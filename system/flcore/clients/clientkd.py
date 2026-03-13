import copy
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client, load_item, save_item


class clientKD(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.mentee_learning_rate = args.mentee_learning_rate
        self.energy = args.T_start
        self.distill_ratio = getattr(args, "distill_ratio", 1.0)  # 每个客户端本地用于KD/DKD的样本比例
        self.distill_type = getattr(args, "distill_type", "KD")   # 蒸馏类型：KD或DKD
        self.dkd_alpha = getattr(args, "dkd_alpha", 1.0)          # DKD中target部分权重
        self.dkd_beta = getattr(args, "dkd_beta", 1.0)           # DKD中non-target部分权重
        self.distill_T = getattr(args, "distill_T", 1.0)         # KD/DKD中logits的温度系数

        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            W_h = nn.Linear(args.feature_dim, args.feature_dim, bias=False).to(self.device)
            save_item(W_h, self.role, 'W_h', self.save_folder_name)
            global_model = load_item('Server', 'global_model', self.save_folder_name)
            save_item(global_model, self.role, 'global_model', self.save_folder_name)

        self.KL = nn.KLDivLoss()
        self.MSE = nn.MSELoss()


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

                    # 特征MSE对齐：在整个batch上计算，不受distill_ratio控制
                    mse_feat = F.mse_loss(rep, W_h(rep_g))

                    # 显式写成 L_d + L_h 的形式，便于和原始FedKD对应
                    denom = (CE_loss + CE_loss_g)
                    L_d = kl_s_t / denom
                    L_d_g = kl_t_s / denom
                    L_h = mse_feat / denom
                    L_h_g = mse_feat / denom

                    distill_loss = L_d + L_h
                    distill_loss_g = L_d_g + L_h_g

                elif self.distill_type == "DKD":
                    # DKD：按照 DKD.py，将 logits 的目标类与非目标类解耦蒸馏，并进行互蒸馏
                    T = self.distill_T
                    alpha = self.dkd_alpha
                    beta = self.dkd_beta

                    num_classes = output.size(1)
                    gt_mask = F.one_hot(y, num_classes=num_classes).bool()
                    other_mask = ~gt_mask

                    # student -> teacher DKD
                    p_s = F.softmax(output / T, dim=1)
                    p_t = F.softmax(output_g / T, dim=1)

                    p_s_t = (p_s * gt_mask).sum(dim=1, keepdim=True)
                    p_s_o = (p_s * other_mask).sum(dim=1, keepdim=True)
                    ps_2 = torch.cat([p_s_t, p_s_o], dim=1)

                    p_t_t = (p_t * gt_mask).sum(dim=1, keepdim=True)
                    p_t_o = (p_t * other_mask).sum(dim=1, keepdim=True)
                    pt_2 = torch.cat([p_t_t, p_t_o], dim=1)

                    log_ps_2 = torch.log(ps_2 + 1e-8)
                    tckd_per_s = F.kl_div(log_ps_2, pt_2, reduction="none").sum(dim=1) * (T * T)

                    logits_t_part2 = output_g / T - 1000.0 * gt_mask.float()
                    logits_s_part2 = output / T - 1000.0 * gt_mask.float()
                    pt_part2 = F.softmax(logits_t_part2, dim=1)
                    log_ps_part2 = F.log_softmax(logits_s_part2, dim=1)
                    nckd_per_s = F.kl_div(log_ps_part2, pt_part2, reduction="none").sum(dim=1) * (T * T)

                    # 样本级 mask 加权（支持 distill_ratio）
                    weight = mask / mask.sum()
                    tckd_loss_s = (tckd_per_s * weight).sum()
                    nckd_loss_s = (nckd_per_s * weight).sum()
                    dkd_raw_s = alpha * tckd_loss_s + beta * nckd_loss_s

                    # teacher -> student DKD（互蒸馏）
                    p_s_rev = F.softmax(output_g / T, dim=1)
                    p_t_rev = F.softmax(output / T, dim=1)

                    p_s_t_rev = (p_s_rev * gt_mask).sum(dim=1, keepdim=True)
                    p_s_o_rev = (p_s_rev * other_mask).sum(dim=1, keepdim=True)
                    ps_2_rev = torch.cat([p_s_t_rev, p_s_o_rev], dim=1)

                    p_t_t_rev = (p_t_rev * gt_mask).sum(dim=1, keepdim=True)
                    p_t_o_rev = (p_t_rev * other_mask).sum(dim=1, keepdim=True)
                    pt_2_rev = torch.cat([p_t_t_rev, p_t_o_rev], dim=1)

                    log_ps_2_rev = torch.log(ps_2_rev + 1e-8)
                    tckd_per_g = F.kl_div(log_ps_2_rev, pt_2_rev, reduction="none").sum(dim=1) * (T * T)

                    logits_t_part2_rev = output / T - 1000.0 * gt_mask.float()
                    logits_s_part2_rev = output_g / T - 1000.0 * gt_mask.float()
                    pt_part2_rev = F.softmax(logits_t_part2_rev, dim=1)
                    log_ps_part2_rev = F.log_softmax(logits_s_part2_rev, dim=1)
                    nckd_per_g = F.kl_div(log_ps_part2_rev, pt_part2_rev, reduction="none").sum(dim=1) * (T * T)

                    tckd_loss_g = (tckd_per_g * weight).sum()
                    nckd_loss_g = (nckd_per_g * weight).sum()
                    dkd_raw_g = alpha * tckd_loss_g + beta * nckd_loss_g

                    # 特征 MSE 对齐：在整个 batch 上计算
                    mse_feat = F.mse_loss(rep, W_h(rep_g))

                    # FedKD 风格的缩放
                    denom = (CE_loss + CE_loss_g)
                    L_d = dkd_raw_s / denom
                    L_d_g = dkd_raw_g / denom
                    L_h = mse_feat / denom
                    L_h_g = mse_feat / denom

                    distill_loss = L_d + L_h
                    distill_loss_g = L_d_g + L_h_g

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

                    mse_feat = F.mse_loss(rep, W_h(rep_g))

                    denom = (CE_loss + CE_loss_g)
                    L_d = kl_s_t / denom
                    L_h = mse_feat / denom

                    distill_loss = L_d + L_h

                elif self.distill_type == "DKD":
                    T = self.distill_T
                    alpha = self.dkd_alpha
                    beta = self.dkd_beta

                    num_classes = output.size(1)
                    gt_mask = F.one_hot(y, num_classes=num_classes).bool()
                    other_mask = ~gt_mask

                    p_s = F.softmax(output / T, dim=1)
                    p_t = F.softmax(output_g / T, dim=1)

                    p_s_t = (p_s * gt_mask).sum(dim=1, keepdim=True)
                    p_s_o = (p_s * other_mask).sum(dim=1, keepdim=True)
                    ps_2 = torch.cat([p_s_t, p_s_o], dim=1)

                    p_t_t = (p_t * gt_mask).sum(dim=1, keepdim=True)
                    p_t_o = (p_t * other_mask).sum(dim=1, keepdim=True)
                    pt_2 = torch.cat([p_t_t, p_t_o], dim=1)

                    log_ps_2 = torch.log(ps_2 + 1e-8)
                    tckd_per = F.kl_div(log_ps_2, pt_2, reduction="none").sum(dim=1) * (T * T)

                    logits_t_part2 = output_g / T - 1000.0 * gt_mask.float()
                    logits_s_part2 = output / T - 1000.0 * gt_mask.float()
                    pt_part2 = F.softmax(logits_t_part2, dim=1)
                    log_ps_part2 = F.log_softmax(logits_s_part2, dim=1)
                    nckd_per = F.kl_div(log_ps_part2, pt_part2, reduction="none").sum(dim=1) * (T * T)

                    weight = mask / mask.sum()
                    tckd_loss = (tckd_per * weight).sum()
                    nckd_loss = (nckd_per * weight).sum()

                    dkd_raw = alpha * tckd_loss + beta * nckd_loss

                    mse_feat = F.mse_loss(rep, W_h(rep_g))

                    denom = (CE_loss + CE_loss_g)
                    L_d = dkd_raw / denom
                    L_h = mse_feat / denom

                    distill_loss = L_d + L_h

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
