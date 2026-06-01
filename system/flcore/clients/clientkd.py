import copy
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client, load_item, save_item


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def _cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdim=True)
    t2 = (t * mask2).sum(dim=1, keepdim=True)
    return torch.cat([t1, t2], dim=1)


def _refine_as_not_true(logits, targets, num_classes):
    nt_positions = torch.arange(0, num_classes, device=logits.device)
    nt_positions = nt_positions.repeat(logits.size(0), 1)
    nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
    nt_positions = nt_positions.view(-1, num_classes - 1)
    return torch.gather(logits, 1, nt_positions)


def dkd_loss_fn(student_logits, teacher_logits, targets, temperature, t_weight, n_weight):
    """DKD loss aligned with federatedlearning/client/kd_dkd_client.py."""
    batch_size = targets.shape[0]
    gt_mask = _get_gt_mask(student_logits, targets)
    other_mask = _get_other_mask(student_logits, targets)
    pred_student = F.softmax(student_logits / temperature, dim=1)
    pred_teacher = F.softmax(teacher_logits / temperature, dim=1)
    pred_student = _cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = _cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    loss_tckd = (
        F.kl_div(log_pred_student, pred_teacher, reduction="sum")
        * (temperature**2)
        / batch_size
    )
    teacher_other_logits = _refine_as_not_true(
        teacher_logits, targets, teacher_logits.size(1),
    )
    student_other_logits = _refine_as_not_true(
        student_logits, targets, student_logits.size(1),
    )
    pred_teacher_part2 = F.softmax(teacher_other_logits / temperature, dim=1)
    log_pred_student_part2 = F.log_softmax(student_other_logits / temperature, dim=1)
    loss_nckd = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction="sum")
        * (temperature**2)
        / batch_size
    )
    return t_weight * loss_tckd + n_weight * loss_nckd


class clientKD(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.mentee_learning_rate = args.mentee_learning_rate
        self.energy = args.T_start
        self.distill_ratio = getattr(args, "distill_ratio", 1.0)  # 每个客户端本地用于KD/DKD的样本比例
        self.distill_type = getattr(args, "distill_type", "KD")   # 蒸馏类型：KD或DKD
        _tckd = getattr(args, "tckd_weight", -1.0)
        _nckd = getattr(args, "nckd_weight", -1.0)
        self.tckd_weight = args.dkd_alpha if _tckd < 0 else _tckd
        self.nckd_weight = args.dkd_beta if _nckd < 0 else _nckd
        self.dkd_weight = getattr(args, "dkd_weight", 1.0)
        self.distill_T = getattr(args, "distill_T", 4.0)
        self.warmup_rounds = getattr(args, "warmup_rounds", 20)
        self.global_round_idx = 0

        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            W_h = nn.Linear(args.feature_dim, args.feature_dim, bias=False).to(self.device)
            save_item(W_h, self.role, 'W_h', self.save_folder_name)
            global_model = load_item('Server', 'global_model', self.save_folder_name)
            save_item(global_model, self.role, 'global_model', self.save_folder_name)

        self.KL = nn.KLDivLoss()
        self.MSE = nn.MSELoss()

    def _compute_dkd_coeff(self, global_round):
        r = max(1, int(global_round))
        if self.warmup_rounds <= 0:
            return 1.0
        return float(min(r, self.warmup_rounds)) / float(self.warmup_rounds)

    def train(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_model = load_item(self.role, 'global_model', self.save_folder_name)
        W_h = load_item(self.role, 'W_h', self.save_folder_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        optimizer_g = torch.optim.SGD(global_model.parameters(), lr=self.mentee_learning_rate)
        optimizer_W = torch.optim.SGD(W_h.parameters(), lr=self.learning_rate)
        # model.to(self.device)
        model.train()
        global_model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = model.base(x)
                rep_g = global_model.base(x)
                output = model.head(rep)          # 学生模型logits
                output_g = global_model.head(rep_g)  # 教师/全局模型logits

                CE_loss = self.loss(output, y)       # 学生的监督损失
                CE_loss_g = self.loss(output_g, y)   # 教师的监督损失

                # 构造蒸馏样本mask：为1的位置参与KD/DKD，为0的位置只做CE
                if self.distill_ratio >= 1.0:
                    mask = torch.ones_like(y, dtype=torch.float32, device=self.device)
                else:
                    mask = (torch.rand_like(y, dtype=torch.float32, device=self.device) < self.distill_ratio).float()

                # 避免极端情况下没有任何样本被选中，保证至少有一个样本参与蒸馏
                if mask.sum() < 1:
                    rand_idx = torch.randint(0, y.shape[0], (1,), device=self.device)
                    mask[rand_idx] = 1.0

                # 蒸馏损失初始化
                distill_loss = torch.tensor(0.0, device=self.device)
                distill_loss_g = torch.tensor(0.0, device=self.device)

                # KD 和 DKD 都在logits层面做蒸馏，并在特征层面做MSE对齐
                if self.distill_type == "KD":
                    # 标准KD：对logits做KL散度蒸馏
                    T = self.distill_T
                    log_p_s = F.log_softmax(output / T, dim=1)
                    p_t = F.softmax(output_g / T, dim=1)
                    log_p_t = F.log_softmax(output_g / T, dim=1)
                    p_s = F.softmax(output / T, dim=1)

                    # 按样本求KL，之后用mask做样本级筛选
                    kl_s_t = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=1)
                    kl_t_s = F.kl_div(log_p_t, p_s, reduction="none").sum(dim=1)

                    kl_s_t = (kl_s_t * mask).sum() / mask.sum()
                    kl_t_s = (kl_t_s * mask).sum() / mask.sum()

                    # 特征MSE对齐，同样只在被选为蒸馏的样本上约束
                    mse_feat = F.mse_loss(rep, W_h(rep_g), reduction="none").mean(dim=1)
                    mse_feat = (mse_feat * mask).sum() / mask.sum()

                    # 显式写成 L_d + L_h 的形式，便于和原始FedKD对应
                    denom = (CE_loss + CE_loss_g)
                    L_d = kl_s_t / denom
                    L_d_g = kl_t_s / denom
                    L_h = mse_feat / denom
                    L_h_g = mse_feat / denom

                    distill_loss = L_d + L_h
                    distill_loss_g = L_d_g + L_h_g

                elif self.distill_type == "DKD":
                    T = self.distill_T
                    student_dkd = dkd_loss_fn(
                        output, output_g, y, T, self.tckd_weight, self.nckd_weight,
                    )
                    teacher_dkd = dkd_loss_fn(
                        output_g, output, y, T, self.tckd_weight, self.nckd_weight,
                    )
                    dkd_coeff = self._compute_dkd_coeff(self.global_round_idx)
                    distill_loss = self.dkd_weight * dkd_coeff * student_dkd
                    distill_loss_g = self.dkd_weight * dkd_coeff * teacher_dkd

                else:
                    # 未知类型时退化为无蒸馏，仅使用CE损失
                    distill_loss = torch.tensor(0.0, device=self.device)
                    distill_loss_g = torch.tensor(0.0, device=self.device)

                loss = CE_loss + distill_loss
                loss_g = CE_loss_g + distill_loss_g

                optimizer.zero_grad()
                optimizer_g.zero_grad()
                optimizer_W.zero_grad()
                loss.backward(retain_graph=True)
                loss_g.backward()
                # prevent divergency on specifical tasks
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(global_model.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(W_h.parameters(), 10)
                optimizer.step()
                optimizer_g.step()
                optimizer_W.step()

        save_item(model, self.role, 'model', self.save_folder_name)
        save_item(global_model, self.role, 'global_model', self.save_folder_name)
        save_item(W_h, self.role, 'W_h', self.save_folder_name)
        compressed_param = decomposition(global_model.named_parameters(), self.energy)
        save_item(compressed_param, self.role, 'compressed_param', self.save_folder_name)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        
    def set_parameters(self):
        global_model = load_item(self.role, 'global_model', self.save_folder_name)
        compressed_param = load_item('Server', 'compressed_param', self.save_folder_name)
        param = recover(compressed_param)
        for name, old_param in global_model.named_parameters():
            if name in param:
                old_param.data = torch.tensor(param[name], device=self.device).data.clone()
        save_item(global_model, self.role, 'global_model', self.save_folder_name)

    def train_metrics(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_model = load_item(self.role, 'global_model', self.save_folder_name)
        W_h = load_item(self.role, 'W_h', self.save_folder_name)
        # model.to(self.device)
        model.eval()
        global_model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = model.base(x)
                rep_g = global_model.base(x)
                output = model.head(rep)          # 学生模型logits
                output_g = global_model.head(rep_g)  # 教师/全局模型logits

                CE_loss = self.loss(output, y)       # 学生的监督损失
                CE_loss_g = self.loss(output_g, y)   # 教师的监督损失

                # 评估时也按照训练时的蒸馏设置构造loss，方便对比
                if self.distill_ratio >= 1.0:
                    mask = torch.ones_like(y, dtype=torch.float32, device=self.device)
                else:
                    mask = (torch.rand_like(y, dtype=torch.float32, device=self.device) < self.distill_ratio).float()
                if mask.sum() < 1:
                    rand_idx = torch.randint(0, y.shape[0], (1,), device=self.device)
                    mask[rand_idx] = 1.0

                distill_loss = torch.tensor(0.0, device=self.device)

                if self.distill_type == "KD":
                    T = self.distill_T
                    log_p_s = F.log_softmax(output / T, dim=1)
                    p_t = F.softmax(output_g / T, dim=1)
                    kl_s_t = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=1)
                    kl_s_t = (kl_s_t * mask).sum() / mask.sum()

                    mse_feat = F.mse_loss(rep, W_h(rep_g), reduction="none").mean(dim=1)
                    mse_feat = (mse_feat * mask).sum() / mask.sum()

                    denom = (CE_loss + CE_loss_g)
                    L_d = kl_s_t / denom
                    L_h = mse_feat / denom

                    distill_loss = L_d + L_h

                elif self.distill_type == "DKD":
                    T = self.distill_T
                    student_dkd = dkd_loss_fn(
                        output, output_g, y, T, self.tckd_weight, self.nckd_weight,
                    )
                    dkd_coeff = self._compute_dkd_coeff(self.global_round_idx)
                    distill_loss = self.dkd_weight * dkd_coeff * student_dkd

                loss = CE_loss + distill_loss
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
            

def recover(compressed_param):
    for k in compressed_param.keys():
        if len(compressed_param[k]) == 3:
            # use np.matmul to support high-dimensional CNN param
            compressed_param[k] = np.matmul(
                compressed_param[k][0] * compressed_param[k][1][..., None, :], 
                    compressed_param[k][2])
    return compressed_param

    
def decomposition(param_iter, energy):
    compressed_param = {}
    for name, param in param_iter:
        try:
            param_cpu = param.detach().cpu().numpy()
        except:
            param_cpu = param
        # refer to https://github.com/wuch15/FedKD/blob/main/run.py#L187
        if param_cpu.shape[0]>1 and len(param_cpu.shape)>1 and 'embeddings' not in name:
            u, sigma, v = np.linalg.svd(param_cpu, full_matrices=False)
            # support high-dimensional CNN param
            if len(u.shape)==4:
                u = np.transpose(u, (2, 3, 0, 1))
                sigma = np.transpose(sigma, (2, 0, 1))
                v = np.transpose(v, (2, 3, 0, 1))
            threshold=0
            if np.sum(np.square(sigma))==0:
                compressed_param_cpu=param_cpu
            else:
                for singular_value_num in range(len(sigma)):
                    if np.sum(np.square(sigma[:singular_value_num]))>energy*np.sum(np.square(sigma)):
                        threshold=singular_value_num
                        break
                u=u[:, :threshold]
                sigma=sigma[:threshold]
                v=v[:threshold, :]
                # support high-dimensional CNN param
                if len(u.shape)==4:
                    u = np.transpose(u, (2, 3, 0, 1))
                    sigma = np.transpose(sigma, (1, 2, 0))
                    v = np.transpose(v, (2, 3, 0, 1))
                compressed_param_cpu=[u,sigma,v]
        elif 'embeddings' not in name:
            compressed_param_cpu=param_cpu

        compressed_param[name] = compressed_param_cpu
        
    return compressed_param
