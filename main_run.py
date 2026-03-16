# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

import os
import sys
import socket
import torch.multiprocessing
import numpy as np

# multiprocessing sharing strategy
torch.multiprocessing.set_sharing_strategy("file_system")
import warnings

warnings.filterwarnings("ignore")

conf_path = os.getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + "/datasets")
sys.path.append(conf_path + "/backbone")
sys.path.append(conf_path + "/models")

from datasets import Priv_NAMES as DATASET_NAMES
from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args
from datasets import get_prive_dataset
from models import get_model
from utils.training import train
from utils.best_args import best_args
from utils.conf import set_random_seed
import setproctitle
import torch
import uuid
import datetime


def parse_args():
    parser = ArgumentParser(description="F2DC", allow_abbrev=False)

    parser.add_argument("--device_id", type=int, default=0, help="device id")
    parser.add_argument(
        "--communication_epoch",
        type=int,
        default=100,
        help="global communication epoch in FL",
    )
    parser.add_argument(
        "--local_epoch", type=int, default=10, help="local epoch for client"
    )
    parser.add_argument(
        "--parti_num", type=int, default=10, help="number for local clients"
    )

    parser.add_argument("--seed", type=int, default=55, help="random seed")
    parser.add_argument(
        "--rand_dataset", type=bool, default=True, help="random set dataset"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="f2dc",
        help="method name",
        choices=get_all_models(),
    )
    parser.add_argument("--structure", type=str, default="heterogeneity")
    parser.add_argument(
        "--dataset",
        type=str,
        default="fl_pacs",
        choices=DATASET_NAMES,
        help="multi-domain dataset for experiments",
    )

    parser.add_argument("--pri_aug", type=str, default="weak", help="data augmentation")
    parser.add_argument(
        "--online_ratio", type=float, default=1, help="ratio for online clients"
    )
    parser.add_argument(
        "--learning_decay", type=bool, default=False, help="learning rate decay"
    )
    parser.add_argument(
        "--averaing", type=str, default="weight", help="averaging strategy"
    )

    parser.add_argument("--save", type=bool, default=True, help="save model params")
    parser.add_argument("--save_name", type=str, default="save_no", help="save name")

    parser.add_argument(
        "--gum_tau", type=float, default=0.1, help="gumbel concrete distribution"
    )
    parser.add_argument("--tem", type=float, default=0.06, help="DFD sep temperature")
    parser.add_argument("--agg_a", type=float, default=1.0, help="domain-aware agg")
    parser.add_argument("--agg_b", type=float, default=0.4, help="domain-aware agg")
    parser.add_argument(
        "--lambda1", type=float, default=0.8, help="params for DFD loss"
    )
    parser.add_argument(
        "--lambda2", type=float, default=1.0, help="params for DFC loss"
    )

    parser.add_argument("--ma_select", type=str, default="resnet", help="backbone")

    # CPU core intra-op parallelism
    torch.set_num_threads(8)
    add_management_args(parser)
    args = parser.parse_args()

    best = best_args[args.dataset][args.model]

    for key, value in best.items():
        setattr(args, key, value)

    if args.seed is not None:
        set_random_seed(args.seed)

    return args


def main_F2DC(args=None):
    if args is None:
        args = parse_args()

    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    priv_dataset = get_prive_dataset(args)
    # default resnet_dc for f2dc
    backbones_list = priv_dataset.get_backbone(args.parti_num, None, args.model)
    # default f2dc method
    model = get_model(backbones_list, args, priv_dataset.get_transform())
    args.arch = model.nets_list[0].name

    print(
        "{}_{}_{}_{}_{}".format(
            args.model,
            args.parti_num,
            args.dataset,
            args.communication_epoch,
            args.local_epoch,
        )
    )
    setproctitle.setproctitle(
        "{}_{}_{}_{}_{}".format(
            args.model,
            args.parti_num,
            args.dataset,
            args.communication_epoch,
            args.local_epoch,
        )
    )

    domains_acc_list = train(model, priv_dataset, args)

    print(f"Accuracy List {args.model} ({args.dataset}):", domains_acc_list)


if __name__ == "__main__":
    main_F2DC()
