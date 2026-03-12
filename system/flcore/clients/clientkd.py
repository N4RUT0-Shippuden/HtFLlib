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
                    # DKD：将logits分解为target和non-target两部分分别蒸馏
                    T = self.distill_T
                    alpha = self.dkd_alpha
                    beta = self.dkd_beta

                    # 教师和学生的softmax概率
                    p_s = F.softmax(output / T, dim=1)
                    p_t = F.softmax(output_g / T, dim=1)

                    # one-hot标签，用于取出target类
                    one_hot = F.one_hot(y, num_classes=output.size(1)).float()

                    # target部分概率
                    p_s_t = (p_s * one_hot).sum(dim=1, keepdim=True)
                    p_t_t = (p_t * one_hot).sum(dim=1, keepdim=True)

                    # non-target部分概率
                    p_s_nt = p_s * (1.0 - one_hot)
                    p_t_nt = p_t * (1.0 - one_hot)

                    # 对non-target部分重新归一化
                    p_s_nt_sum = p_s_nt.sum(dim=1, keepdim=True) + 1e-8
                    p_t_nt_sum = p_t_nt.sum(dim=1, keepdim=True) + 1e-8
                    p_s_nt_norm = p_s_nt / p_s_nt_sum
                    p_t_nt_norm = p_t_nt / p_t_nt_sum

                    # target部分的DKD项（KL散度）
                    log_p_s_t = torch.log(p_s_t + 1e-8)
                    log_p_t_t = torch.log(p_t_t + 1e-8)
                    dkd_target = F.kl_div(log_p_s_t, p_t_t, reduction="none").view(-1)
                    dkd_target_g = F.kl_div(log_p_t_t, p_s_t, reduction="none").view(-1)

                    # non-target部分的DKD项（KL散度）
                    log_p_s_nt = torch.log(p_s_nt_norm + 1e-8)
                    log_p_t_nt = torch.log(p_t_nt_norm + 1e-8)
                    dkd_non = F.kl_div(log_p_s_nt, p_t_nt_norm, reduction="none").sum(dim=1)
                    dkd_non_g = F.kl_div(log_p_t_nt, p_s_nt_norm, reduction="none").sum(dim=1)

                    # 应用样本级mask
                    dkd_target = (dkd_target * mask).sum() / mask.sum()
                    dkd_target_g = (dkd_target_g * mask).sum() / mask.sum()
                    dkd_non = (dkd_non * mask).sum() / mask.sum()
                    dkd_non_g = (dkd_non_g * mask).sum() / mask.sum()

                    # 特征MSE对齐：在整个batch上计算，不受distill_ratio控制
                    mse_feat = F.mse_loss(rep, W_h(rep_g))

                    # DKD情况下，同样拆成 L_d（由target和non-target两部分组成）和 L_h
                    denom = (CE_loss + CE_loss_g)
                    L_d = (alpha * dkd_target + beta * dkd_non) / denom
                    L_d_g = (alpha * dkd_target_g + beta * dkd_non_g) / denom
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

                    p_s = F.softmax(output / T, dim=1)
                    p_t = F.softmax(output_g / T, dim=1)
                    one_hot = F.one_hot(y, num_classes=output.size(1)).float()

                    p_s_t = (p_s * one_hot).sum(dim=1, keepdim=True)
                    p_t_t = (p_t * one_hot).sum(dim=1, keepdim=True)

                    p_s_nt = p_s * (1.0 - one_hot)
                    p_t_nt = p_t * (1.0 - one_hot)

                    p_s_nt_sum = p_s_nt.sum(dim=1, keepdim=True) + 1e-8
                    p_t_nt_sum = p_t_nt.sum(dim=1, keepdim=True) + 1e-8
                    p_s_nt_norm = p_s_nt / p_s_nt_sum
                    p_t_nt_norm = p_t_nt / p_t_nt_sum

                    log_p_s_t = torch.log(p_s_t + 1e-8)
                    log_p_t_t = torch.log(p_t_t + 1e-8)
                    dkd_target = F.kl_div(log_p_s_t, p_t_t, reduction="none").view(-1)

                    log_p_s_nt = torch.log(p_s_nt_norm + 1e-8)
                    dkd_non = F.kl_div(log_p_s_nt, p_t_nt_norm, reduction="none").sum(dim=1)

                    dkd_target = (dkd_target * mask).sum() / mask.sum()
                    dkd_non = (dkd_non * mask).sum() / mask.sum()

                    mse_feat = F.mse_loss(rep, W_h(rep_g))

                    denom = (CE_loss + CE_loss_g)
                    L_d = (alpha * dkd_target + beta * dkd_non) / denom
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
    """对全局模型参数做 SVD 压缩；若出现数值问题（NaN/Inf 或 SVD 不收敛），则退回使用原参数，避免训练直接崩溃。"""
    compressed_param = {}
    for name, param in param_iter:
        try:
            # 原始行为：把张量搬到 CPU 并转成 numpy，便于后面用 np.linalg.svd 处理
            param_cpu = param.detach().cpu().numpy()
        except:
            # 保险起见：如果 detach/cpu 失败，就直接使用原对象
            param_cpu = param

        # 新行为：默认先使用“原参数”作为压缩结果（退回原参数的含义）
        compressed_param_cpu = param_cpu

        # 只对形状合适、且名字中不包含 'embeddings' 的参数做 SVD 压缩
        if (
            isinstance(param_cpu, np.ndarray)
            and param_cpu.ndim > 1
            and param_cpu.shape[0] > 1
            and 'embeddings' not in name
        ):
            # 新增安全检查：如果该层参数中存在 NaN 或 Inf，直接跳过压缩，避免数值不稳定
            if not np.all(np.isfinite(param_cpu)):
                compressed_param[name] = compressed_param_cpu
                continue

            # 新增容错：SVD 可能抛出 "SVD did not converge"，此时退回原参数而不是让程序崩溃
            try:
                u, sigma, v = np.linalg.svd(param_cpu, full_matrices=False)
            except np.linalg.LinAlgError:
                compressed_param[name] = compressed_param_cpu
                continue

            # 原有行为：支持高维 CNN 参数，把维度重排为适合后续处理的形式
            if len(u.shape) == 4:
                u = np.transpose(u, (2, 3, 0, 1))
                sigma = np.transpose(sigma, (2, 0, 1))
                v = np.transpose(v, (2, 3, 0, 1))

            threshold = 0
            # 新增变量：提前计算总能量，便于多次复用
            total_energy = np.sum(np.square(sigma))
            if total_energy == 0:
                # 极端情况：所有奇异值能量为 0，压缩没有意义，退回原参数
                compressed_param_cpu = param_cpu
            else:
                # 原有逻辑：找到能量累积超过 energy 比例的最小奇异值个数
                for singular_value_num in range(len(sigma)):
                    if np.sum(np.square(sigma[:singular_value_num])) > energy * total_energy:
                        threshold = singular_value_num
                        break

                if threshold == 0:
                    # 若 threshold 仍为 0，说明无法选出有效子空间，退回原参数
                    compressed_param_cpu = param_cpu
                else:
                    # 截断奇异值分解结果，只保留前 threshold 个奇异值
                    u = u[:, :threshold]
                    sigma = sigma[:threshold]
                    v = v[:threshold, :]

                    # 原有行为：对高维 CNN 参数再做一次维度重排
                    if len(u.shape) == 4:
                        u = np.transpose(u, (2, 3, 0, 1))
                        sigma = np.transpose(sigma, (1, 2, 0))
                        v = np.transpose(v, (2, 3, 0, 1))

                    # 新行为：真正成功压缩时，才把结果存成 [u, sigma, v]
                    compressed_param_cpu = [u, sigma, v]

        # 无论是否压缩成功，都把当前（可能是压缩后的，也可能是原始的）参数写入字典
        compressed_param[name] = compressed_param_cpu

    return compressed_param
