import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel
import torch


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated Learning F2DC')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser

def get_pred(out, labels): 
    pred = out.sort(dim=-1, descending=True)[1][:, 0]
    second_pred = out.sort(dim=-1, descending=True)[1][:, 1]
    wrong_high_label = torch.where(pred == labels, second_pred, pred)
    return wrong_high_label


class F2DC(FederatedModel):
    NAME = 'f2dc'
    COMPATIBILITY = ['heterogeneity']

    def __init__(self, nets_list, args, transform):
        super(F2DC, self).__init__(nets_list, args, transform)
        self.args = args
        self.tem = self.args.tem

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients
        all_clients_loss = 0.0
        self.num_samples = []

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

        num_c_samples = 0
        iterator = tqdm(range(self.local_epoch))
        global_loss = 0.0
        global_samples = 0

        for iter in iterator:
            epoch_loss = 0.0
            epoch_samples = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                images = images.to(self.device)
                labels = labels.to(self.device)

                out, feat, ro_outputs, re_outputs, rec_outputs, ro_flatten, re_flatten = net(images)
                outputs = out
                wrong_high_labels = get_pred(out, labels)

                DFD_dis1_loss = torch.tensor(0.).to(self.device)
                if not len(ro_outputs) == 0:
                    for ro_out in ro_outputs:
                        DFD_dis1_loss += 1.0 * criterion(ro_out, labels)
                    DFD_dis1_loss /= len(ro_outputs)
                DFD_dis2_loss = torch.tensor(0.).to(self.device)
                if not len(re_outputs) == 0:
                    for re_out in re_outputs:
                        DFD_dis2_loss += 1.0 * criterion(re_out, wrong_high_labels)
                    DFD_dis2_loss /= len(re_outputs)
                DFD_sep_loss = torch.tensor(0.).to(self.device)
                l_cos = torch.cosine_similarity(ro_flatten, re_flatten, dim=1)
                l_cos = l_cos / self.tem
                exp_l_cos = torch.exp(l_cos)
                DFD_sep_loss += torch.log(exp_l_cos)
                DFD_sep_loss /= ro_flatten.size(0)
                
                DFD_loss = DFD_dis1_loss + DFD_dis2_loss + DFD_sep_loss

                DFC_loss = torch.tensor(0.).to(self.device)
                if not len(rec_outputs) == 0:
                    for rec_out in rec_outputs:
                        DFC_loss += 1.0 * criterion(rec_out, labels)
                    DFC_loss /= len(rec_outputs)

                loss_DC = self.args.lambda1 * DFD_loss + self.args.lambda2 * DFC_loss 
                loss_CE = criterion(outputs, labels)
                loss = loss_CE + loss_DC

                loss.backward()
                iterator.desc = "Local Pariticipant %d DC = %0.3f, CE = %0.3f" % (index, loss_DC, loss_CE)
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