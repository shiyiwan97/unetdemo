from config import Config
from utils.loss_util import LossUtil
from utils.optimizer_util import OptimizerUtil


def get_config():
    train_data_path = r'F:\machineLearning\dataset\test\train'
    test_data_path = r'F:\machineLearning\dataset\test\test'
    log_dir = r'F:\machineLearning\dataset\test\log'
    weight_path = r'weight\weight_latest.pth'
    load_weight = 0
    loss_function = LossUtil.FocalLoss
    optimizer = OptimizerUtil.SGD

    return Config(train_data_path, test_data_path, log_dir, weight_path, load_weight, loss_function, optimizer)
