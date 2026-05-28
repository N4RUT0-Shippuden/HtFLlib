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
        self.distill_ratio = getattr(args, "distill_ratio", 1.0)  # 每个客户端本地用于KD/DKD的样本比例
        self.distill_type = getattr(args, "distill_type", "KD")   # 蒸馏类型：KD或DKD
        self.dkd_alpha = getattr(args, "dkd_alpha", 1.0)          # DKD中target部分权重
        self.dkd_beta = getattr(args, "dkd_beta", 1.0)           # DKD中non-target部分权重
        self.distill_T = getattr(args, "distill_T", 4.0)         # KD/DKD中logits的温度系数
        self.warmup_rounds = getattr(args, "warmup_rounds", 20)  # DKD蒸馏项线性warmup轮数
        self.global_round_idx = 0

        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            global_model = load_item('Server', 'global_model', self.save_folder_name)
            save_item(global_model, self.role, 'global_model', self.save_folder_name)

    def _compute_dkd_coeff(self, global_round):
        r = max(1, int(global_round))
        if self.warmup_rounds <= 0:
            return 1.0
        return float(min(r, self.warmup_rounds)) / float(self.warmup_rounds)

    def train(self):
        trainloader = self.load_train_data()
        if self.distill_ratio < 1.0:
            dataset = trainloader.dataset
            total_num = len(dataset)
            selected_num = max(1, int(self.distill_ratio * total_num))
            indices = torch.randperm(total_num)[:selected_num].tolist()
            subset = torch.utils.data.Subset(dataset, indices)
            trainloader = torch.utils.data.DataLoader(
                subset,
                batch_size=trainloader.batch_size,
                shuffle=False,
                drop_last=trainloader.drop_last,
            )
        model = load_item(self.role, 'model', self.save_folder_name)
        global_model = load_item(self.role, 'global_model', self.save_folder_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        optimizer_g = torch.optim.SGD(global_model.parameters(), lr=self.mentee_learning_rate)
        # model.to(self.device)
        model.train()
        global_model.train()
        
        start_time = time.time()
        dkd_coeff = self._compute_dkd_coeff(self.global_round_idx) if self.distill_type == "DKD" else 1.0

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

                distill_loss = torch.tensor(0.0, device=self.device)
                distill_loss_g = torch.tensor(0.0, device=self.device)

                # KD 和 DKD 都在logits层面做蒸馏（不做特征对齐）
                if self.distill_type == "KD":
                    # 标准KD：对logits做KL散度蒸馏
                    T = self.distill_T
                    log_p_s = F.log_softmax(output / T, dim=1)
                    p_t = F.softmax(output_g / T, dim=1)
                    log_p_t = F.log_softmax(output_g / T, dim=1)
                    p_s = F.softmax(output / T, dim=1)

                    kl_s_t = F.kl_div(log_p_s, p_t, reduction="batchmean")
                    kl_t_s = F.kl_div(log_p_t, p_s, reduction="batchmean")

                    denom = (CE_loss + CE_loss_g)
                    L_d = kl_s_t / denom
                    L_d_g = kl_t_s / denom

                    distill_loss = L_d
                    distill_loss_g = L_d_g

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

                    dkd_target = dkd_target.mean()
                    dkd_target_g = dkd_target_g.mean()
                    dkd_non = dkd_non.mean()
                    dkd_non_g = dkd_non_g.mean()

                    denom = (CE_loss + CE_loss_g)
                    L_d = (alpha * dkd_target + beta * dkd_non) / denom
                    L_d_g = (alpha * dkd_target_g + beta * dkd_non_g) / denom

                    distill_loss = dkd_coeff * L_d
                    distill_loss_g = dkd_coeff * L_d_g

                else:
                    # 未知类型时退化为无蒸馏，仅使用CE损失
                    distill_loss = torch.tensor(0.0, device=self.device)
                    distill_loss_g = torch.tensor(0.0, device=self.device)

                loss = CE_loss + distill_loss
                loss_g = CE_loss_g + distill_loss_g

                optimizer.zero_grad()
                optimizer_g.zero_grad()
                loss.backward(retain_graph=True)
                loss_g.backward()
                # prevent divergency on specifical tasks
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(global_model.parameters(), 10)
                optimizer.step()
                optimizer_g.step()

        save_item(model, self.role, 'model', self.save_folder_name)
        save_item(global_model, self.role, 'global_model', self.save_folder_name)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        
    def set_parameters(self):
        global_model = load_item(self.role, 'global_model', self.save_folder_name)
        server_global_model = load_item('Server', 'global_model', self.save_folder_name)
        global_model.load_state_dict(server_global_model.state_dict(), strict=True)
        save_item(global_model, self.role, 'global_model', self.save_folder_name)

    def train_metrics(self):
        trainloader = self.load_train_data()
        if self.distill_ratio < 1.0:
            dataset = trainloader.dataset
            total_num = len(dataset)
            selected_num = max(1, int(self.distill_ratio * total_num))
            indices = torch.randperm(total_num)[:selected_num].tolist()
            subset = torch.utils.data.Subset(dataset, indices)
            trainloader = torch.utils.data.DataLoader(
                subset,
                batch_size=trainloader.batch_size,
                shuffle=False,
                drop_last=trainloader.drop_last,
            )
        model = load_item(self.role, 'model', self.save_folder_name)
        global_model = load_item(self.role, 'global_model', self.save_folder_name)
        # model.to(self.device)
        model.eval()
        global_model.eval()

        train_num = 0
        losses = 0
        dkd_coeff = self._compute_dkd_coeff(self.global_round_idx) if self.distill_type == "DKD" else 1.0
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

                distill_loss = torch.tensor(0.0, device=self.device)

                if self.distill_type == "KD":
                    T = self.distill_T
                    log_p_s = F.log_softmax(output / T, dim=1)
                    p_t = F.softmax(output_g / T, dim=1)
                    kl_s_t = F.kl_div(log_p_s, p_t, reduction="batchmean")

                    denom = (CE_loss + CE_loss_g)
                    L_d = kl_s_t / denom

                    distill_loss = L_d

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

                    dkd_target = dkd_target.mean()
                    dkd_non = dkd_non.mean()

                    denom = (CE_loss + CE_loss_g)
                    L_d = (alpha * dkd_target + beta * dkd_non) / denom

                    distill_loss = dkd_coeff * L_d

                loss = CE_loss + distill_loss
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
