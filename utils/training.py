import torch
from argparse import Namespace
from models.utils.federated_model import FederatedModel
from datasets.utils.federated_dataset import FederatedDataset
from typing import Tuple
from torch.utils.data import DataLoader
import numpy as np
from utils.logger import CsvWriter
from collections import Counter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from backbone.ResNet import resnet10, resnet12
import time


def global_evaluate(
    model: FederatedModel, test_dl: DataLoader, setting: str, name: str
) -> Tuple[list, list]:
    accs = []
    net = model.global_net
    status = net.training
    net.eval()
    for j, dl in enumerate(test_dl):
        correct, total, top1, top5 = 0.0, 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(dl):
            with torch.no_grad():
                images, labels = images.to(model.device), labels.to(model.device)
                if model.NAME == "f2dc":
                    outputs, _, _, _, _ = net(images)
                else:
                    outputs = net(images)
                _, max5 = torch.topk(outputs, 5, dim=-1)
                labels = labels.view(-1, 1)
                top1 += (labels == max5[:, 0:1]).sum().item()
                top5 += (labels == max5).sum().item()
                total += labels.size(0)
        top1acc = round(100 * top1 / total, 2)
        top5acc = round(100 * top5 / total, 2)
        accs.append(top1acc)
    net.train(status)
    return accs


def get_prototypes(features, labels):
    centers = []
    for i in range(10):
        idx = labels == i
        class_feat = features[idx]
        center = np.mean(class_feat, axis=0)
        centers.append(center)
    centers = np.array(centers)
    return centers


def get_features(net, dataloader, device):
    net.eval()
    features, labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            feat = net.features(x)  # 512
            features.append(feat.cpu())
            labels.append(y.cpu())
    features = torch.cat(features, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return features, labels


def extract_features(model, dataloader):
    net = model.global_net
    net.eval()
    features, labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(model.device), y.to(model.device)
            feat = net.features(x)  # 512
            features.append(feat.cpu())
            labels.append(y.cpu())
    features = torch.cat(features, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return features, labels


def train(
    model: FederatedModel, private_dataset: FederatedDataset, args: Namespace
) -> None:
    if args.csv_log:
        csv_writer = CsvWriter(args, private_dataset)
    model.N_CLASS = private_dataset.N_CLASS
    domains_list = private_dataset.DOMAINS_LIST
    domains_len = len(domains_list)

    if args.rand_dataset:
        max_num = 10
        is_ok = False
        while not is_ok:
            if model.args.dataset == "fl_officecaltech":
                domains_list = ["caltech", "amazon", "webcam", "dslr"]
                selected_domain_list = np.random.choice(
                    domains_list,
                    size=args.parti_num - domains_len,
                    replace=True,
                    p=None,
                )
                selected_domain_list = list(selected_domain_list) + domains_list
            elif model.args.dataset == "fl_digits":
                domains_list = ["mnist", "usps", "svhn", "syn"]
                selected_domain_list = np.random.choice(
                    domains_list, size=args.parti_num, replace=True, p=None
                )
            elif model.args.dataset == "fl_pacs":
                domains_list = ["photo", "art", "cartoon", "sketch"]
                selected_domain_list = np.random.choice(
                    domains_list, size=args.parti_num, replace=True, p=None
                )

            result = dict(Counter(selected_domain_list))

            for k in result:
                if result[k] > max_num:
                    is_ok = False
                    break
            else:
                is_ok = True

    else:
        selected_domain_dict = {
            "mnist": 6,
            "usps": 4,
            "svhn": 3,
            "syn": 7,
        }  # base for fl_digits
        selected_domain_list = []
        for k in selected_domain_dict:
            domain_num = selected_domain_dict[k]
            for i in range(domain_num):
                selected_domain_list.append(k)

        selected_domain_list = np.random.permutation(selected_domain_list)

        result = Counter(selected_domain_list)

    # print(result)
    print(f"selected_domain_list for {args.parti_num} clients as:")
    print(selected_domain_list)
    pri_train_loaders, test_loaders = private_dataset.get_data_loaders(
        selected_domain_list
    )
    model.trainloaders = pri_train_loaders
    model.testlodaers = test_loaders

    if hasattr(model, "ini"):
        model.ini()

    accs_dict = {}
    mean_accs_list = []
    all_l2_dis = []
    best_mean_acc = 0.0

    if model.args.dataset == "fl_officecaltech":
        all_dataset_names = ["caltech", "amazon", "webcam", "dslr"]
    elif model.args.dataset == "fl_digits":
        all_dataset_names = ["mnist", "usps", "svhn", "syn"]
    elif model.args.dataset == "fl_pacs":
        all_dataset_names = ["photo", "art", "cartoon", "sketch"]

    Epoch = args.communication_epoch
    all_epoch_loss = []
    for epoch_index in range(Epoch):
        model.epoch_index = epoch_index

        start_time = time.time()
        if hasattr(model, "loc_update"):
            epoch_loss = model.loc_update(pri_train_loaders)
            all_epoch_loss.append(epoch_loss)
        end_time = time.time()
        print(
            "The " + str(epoch_index) + " Communcation Time:",
            round(end_time - start_time, 3),
        )

        # all_dis = 0.0

        accs = global_evaluate(
            model, test_loaders, private_dataset.SETTING, private_dataset.NAME
        )
        mean_acc = round(np.mean(accs, axis=0), 3)
        mean_accs_list.append(mean_acc)
        for i in range(len(accs)):
            if i in accs_dict:
                accs_dict[i].append(accs[i])
            else:
                accs_dict[i] = [accs[i]]

        print(
            "The " + str(epoch_index) + " Communcation Accuracy:",
            str(mean_acc),
            "Method:",
            model.args.model,
            "epoch loss:",
            str(epoch_loss),
        )
        print(accs)
        print()

        if args.save:
            if args.save_name == "No":
                pth_name = args.model
            else:
                pth_name = args.save_name

            if mean_acc > best_mean_acc:
                best_mean_acc = mean_acc

    if args.csv_log:
        csv_writer.write_acc(accs_dict, mean_accs_list)

    print("ALL Loss: ", all_epoch_loss)
    return mean_accs_list
