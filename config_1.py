import os

import torch

from config import Config
from utils.loss_util import LossUtil
from utils.optimizer_util import OptimizerUtil


def get_config():
    train_data_path = r'D:\Dataset\dataset_1000\train'
    test_data_path = r'D:\Dataset\dataset_1000\test'
    log_dir = r'log'
    weight_path = r'weight\weight_latest.pth'
    folder= r'trainFolder'
    load_weight = 0
    # loss_function = LossUtil.FocalLoss(torch.tensor([1, 1, 1]), 2)
    loss_function =LossUtil.CrossEntropyLoss(255)
    optimizer = OptimizerUtil.SGD(0.001, 0.8, 1e-2)

    return Config(train_data_path, test_data_path, log_dir, weight_path, load_weight, loss_function, optimizer)

def getPath():
    os.listdir()
    return
