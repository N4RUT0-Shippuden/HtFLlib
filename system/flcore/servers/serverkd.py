import copy
import random
import time

import numpy as np
from flcore.clients.clientkd import clientKD, recover, decomposition
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from threading import Thread
from flcore.trainmodel.models import BaseHeadSplit


class FedKD(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            if hasattr(args, 'global_model'):
                global_model = BaseHeadSplit(args, is_global=True).to(args.device)
            else:
                global_model = BaseHeadSplit(args, 0).to(args.device)
            save_item(global_model, self.role, 'global_model', self.save_folder_name)
        
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientKD)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.T_start = args.T_start
        self.T_end = args.T_end
        self.energy = self.T_start


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_ids()
            self.aggregate_parameters()

            self.send_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

            self.energy = self.T_start + ((1 + i) / self.global_rounds) * (self.T_end - self.T_start)
            for client in self.clients:
                client.energy = self.energy

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()

        
    def aggregate_parameters(self):
        """聚合客户端上传的全局模型参数，不做 SVD 压缩，直接采用参数平均。"""
        # 初始化全局参数为 0
        global_model = load_item(self.role, 'global_model', self.save_folder_name)
        global_param = {name: torch.zeros_like(param, device='cpu')
                        for name, param in global_model.named_parameters()}

        # 直接从每个客户端加载其保存的 global_model，并做 FedAvg 聚合
        for cid in self.uploaded_ids:
            client = self.clients[cid]
            client_global = load_item(client.role, 'global_model', client.save_folder_name)
            for (name, g_param), (_, c_param) in zip(global_param.items(), client_global.named_parameters()):
                global_param[name] += c_param.detach().cpu() / len(self.uploaded_ids)

        # 将聚合后的参数写回服务器端的 global_model
        for name, param in global_model.named_parameters():
            if name in global_param:
                param.data = global_param[name].to(param.device)

        save_item(global_model, self.role, 'global_model', self.save_folder_name)