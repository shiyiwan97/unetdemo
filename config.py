import datetime
import os
from utils.loss_util import LossUtil
from utils.optimizer_util import OptimizerUtil

from torch.utils.tensorboard import SummaryWriter


class Config:
    """
    配置类：配置各种路径等配置
    """

    def __init__(self, train_data_path, test_data_path, log_dir, weight_path, load_weight, loss_function: LossUtil,
                 optimizer: OptimizerUtil):
        """

        :param train_data_path:
        :param test_data_path:
        :param log_dir:
        :param weight_path:
        :param load_weight: 0=latest,1=max_iou,3=min_loss
        :param loss_function:
        """
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.log_dir = log_dir
        self.testDataPath = test_data_path
        self.testDataPath = test_data_path

        now = datetime.datetime.now()
        now = str(now).replace(':', '.')
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, str(len(os.listdir(log_dir)) + 1) + '.' + str(now)))

        self.weight_path = weight_path
        self.weight_path_latest = os.path.join(self.weight_path, 'latest_weight.pth')
        self.loss_record_path = os.path.join(self.weight_path, 'iou')
        self.weight_path_iou = os.path.join(self.weight_path, 'max_IoU_weight.pth')
        self.iou_record_loss = os.path.join(self.weight_path, 'loss')
        if load_weight == 0:
            self.load_weight_path = self.weight_path_latest
        elif load_weight == 1:
            self.load_weight_path = self.weight_path_iou
        elif load_weight == 2:
            self.load_weight_path = self.iou_record_loss
        else:
            raise Exception('参数错误：load_weight.（0=latest,1=max_iou,3=min_loss）')

        self.loss_function = loss_function
        self.optimizer = optimizer


