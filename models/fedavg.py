import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
import torch
import numpy as np
from utils.args import *
from models.utils.federated_model import FederatedModel


class FedAvG(FederatedModel):
    NAME = 'fedavg'
    COMPATIBILITY = ['heterogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedAvG, self).__init__(nets_list, args, transform)

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _,net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients,self.online_num,replace=False).tolist()
        self.online_clients = online_clients

        self.num_samples = []
        all_clients_loss = 0.0
        for i in online_clients:
            c_loss, c_samples = self._train_net(i, self.nets_list[i], priloader_list[i])
            all_clients_loss += c_loss
            self.num_samples.append(c_samples)

        self.aggregate_nets(None)

        all_c_avg_loss = all_clients_loss / len(online_clients)
        return round(all_c_avg_loss, 3)

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)

        global_loss = 0.0
        global_samples = 0
        num_c_samples = 0

        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            epoch_loss = 0.0
            epoch_samples = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)
                optimizer.step()
                batch_size = labels.size(0)
                epoch_loss += loss.item() * batch_size
                epoch_samples += batch_size
            
            avg_epoch_loss = epoch_loss / epoch_samples
            global_loss += epoch_loss
            global_samples += epoch_samples
            num_c_samples = epoch_samples
        
        global_avg_loss = global_loss / global_samples

        return round(global_avg_loss, 3), num_c_samples 