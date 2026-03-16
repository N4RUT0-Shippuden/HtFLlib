import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from flcore.clients.clientbase import Client, load_item, save_item
from collections import defaultdict


class clientFD(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.lamda = args.lamda
        self.distill_ratio = getattr(args, "distill_ratio", 1.0)
        self.kd_ce_weight = getattr(args, "kd_ce_loss", 1.0)
        self.kd_loss_weight = getattr(args, "kd_loss", 1.0)
        self.dkd_ce_weight = getattr(args, "dkd_ce_loss", 1.0)
        self.dkd_warmup = getattr(args, "dkd_warmup", 20)
        self.distill_type = getattr(args, "distill_type", "KD")
        self.distill_T = getattr(args, "distill_T", 4.0)
        self.dkd_alpha = getattr(args, "dkd_alpha", 1.0)
        self.dkd_beta = getattr(args, "dkd_beta", 8.0)


    def train(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        global_logits = load_item('Server', 'global_logits', self.save_folder_name)
        
        start_time = time.time()

        # model.to(self.device)
        model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        logits = defaultdict(list)
        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = model(x)
                kd_loss = torch.tensor(0.0, device=self.device)

                if self.distill_ratio >= 1.0:
                    mask = torch.ones_like(y, dtype=torch.float32, device=self.device)
                else:
                    mask = (torch.rand_like(y, dtype=torch.float32, device=self.device) < self.distill_ratio).float()
                if mask.sum() < 1:
                    rand_idx = torch.randint(0, y.shape[0], (1,), device=self.device)
                    mask[rand_idx] = 1.0

                ce_per = F.cross_entropy(output, y, reduction="none")
                ce_loss = (ce_per * mask).sum() / mask.sum()

                if global_logits is not None:
                    teacher_logits = copy.deepcopy(output.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(global_logits[y_c]) != type([]):
                            teacher_logits[i, :] = global_logits[y_c].data

                    if self.distill_type == "KD":
                        T = self.distill_T
                        log_p_s = F.log_softmax(output / T, dim=1)
                        p_t = F.softmax(teacher_logits / T, dim=1)
                        kl_per = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=1) * (T * T)
                        kd_loss = (kl_per * mask).sum() / mask.sum()
                    elif self.distill_type == "DKD":
                        T = self.distill_T
                        alpha = self.dkd_alpha
                        beta = self.dkd_beta
                        num_classes = output.size(1)
                        gt_mask = F.one_hot(y, num_classes=num_classes).bool()
                        other_mask = ~gt_mask

                        p_s = F.softmax(output / T, dim=1)
                        p_t = F.softmax(teacher_logits / T, dim=1)

                        p_s_t = (p_s * gt_mask).sum(dim=1, keepdim=True)
                        p_s_o = (p_s * other_mask).sum(dim=1, keepdim=True)
                        ps_2 = torch.cat([p_s_t, p_s_o], dim=1)

                        p_t_t = (p_t * gt_mask).sum(dim=1, keepdim=True)
                        p_t_o = (p_t * other_mask).sum(dim=1, keepdim=True)
                        pt_2 = torch.cat([p_t_t, p_t_o], dim=1)

                        log_ps_2 = torch.log(ps_2 + 1e-8)
                        tckd_per = F.kl_div(log_ps_2, pt_2, reduction="none").sum(dim=1) * (T * T)

                        logits_t_part2 = teacher_logits / T - 1000.0 * gt_mask.float()
                        logits_s_part2 = output / T - 1000.0 * gt_mask.float()
                        pt_part2 = F.softmax(logits_t_part2, dim=1)
                        log_ps_part2 = F.log_softmax(logits_s_part2, dim=1)
                        nckd_per = F.kl_div(log_ps_part2, pt_part2, reduction="none").sum(dim=1) * (T * T)

                        weight = mask / mask.sum()
                        tckd_loss = (tckd_per * weight).sum()
                        nckd_loss = (nckd_per * weight).sum()
                        warmup_factor = 1.0
                        if self.dkd_warmup > 0:
                            warmup_factor = min(float(step + 1) / float(self.dkd_warmup), 1.0)
                        kd_loss = (alpha * tckd_loss + beta * nckd_loss) * warmup_factor

                if self.distill_type == "KD":
                    loss = self.kd_ce_weight * ce_loss + self.kd_loss_weight * kd_loss
                elif self.distill_type == "DKD":
                    loss = self.dkd_ce_weight * ce_loss + self.lamda * kd_loss
                else:
                    loss = ce_loss + kd_loss

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    logits[y_c].append(output[i, :].detach().data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        save_item(model, self.role, 'model', self.save_folder_name)
        save_item(agg_func(logits), self.role, 'logits', self.save_folder_name)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def train_metrics(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_logits = load_item('Server', 'global_logits', self.save_folder_name)
        # model.to(self.device)
        model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)
                ce_loss = self.loss(output, y)
                kd_loss = torch.tensor(0.0, device=self.device)

                if global_logits is not None:
                    teacher_logits = copy.deepcopy(output.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(global_logits[y_c]) != type([]):
                            teacher_logits[i, :] = global_logits[y_c].data

                    if self.distill_type == "KD":
                        T = self.distill_T
                        log_p_s = F.log_softmax(output / T, dim=1)
                        p_t = F.softmax(teacher_logits / T, dim=1)
                        kd_loss = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
                    elif self.distill_type == "DKD":
                        T = self.distill_T
                        alpha = self.dkd_alpha
                        beta = self.dkd_beta
                        num_classes = output.size(1)
                        gt_mask = F.one_hot(y, num_classes=num_classes).bool()
                        other_mask = ~gt_mask

                        p_s = F.softmax(output / T, dim=1)
                        p_t = F.softmax(teacher_logits / T, dim=1)

                        p_s_t = (p_s * gt_mask).sum(dim=1, keepdim=True)
                        p_s_o = (p_s * other_mask).sum(dim=1, keepdim=True)
                        ps_2 = torch.cat([p_s_t, p_s_o], dim=1)

                        p_t_t = (p_t * gt_mask).sum(dim=1, keepdim=True)
                        p_t_o = (p_t * other_mask).sum(dim=1, keepdim=True)
                        pt_2 = torch.cat([p_t_t, p_t_o], dim=1)

                        log_ps_2 = torch.log(ps_2 + 1e-8)
                        tckd_per = F.kl_div(log_ps_2, pt_2, reduction="none").sum(dim=1) * (T * T)

                        logits_t_part2 = teacher_logits / T - 1000.0 * gt_mask.float()
                        logits_s_part2 = output / T - 1000.0 * gt_mask.float()
                        pt_part2 = F.softmax(logits_t_part2, dim=1)
                        log_ps_part2 = F.log_softmax(logits_s_part2, dim=1)
                        nckd_per = F.kl_div(log_ps_part2, pt_part2, reduction="none").sum(dim=1) * (T * T)

                        tckd_loss = tckd_per.mean()
                        nckd_loss = nckd_per.mean()
                        kd_loss = alpha * tckd_loss + beta * nckd_loss

                if self.distill_type == "KD":
                    loss = self.kd_ce_weight * ce_loss + self.kd_loss_weight * kd_loss
                elif self.distill_type == "DKD":
                    loss = self.dkd_ce_weight * ce_loss + self.lamda * kd_loss
                else:
                    loss = ce_loss + kd_loss

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num


# https://github.com/yuetan031/fedlogit/blob/main/lib/utils.py#L205
def agg_func(logits):
    """
    Returns the average of the weights.
    """

    for [label, logit_list] in logits.items():
        if len(logit_list) > 1:
            logit = 0 * logit_list[0].data
            for i in logit_list:
                logit += i.data
            logits[label] = logit / len(logit_list)
        else:
            logits[label] = logit_list[0]

    return logits
