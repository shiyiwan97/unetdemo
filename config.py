import datetime
import os
from utils.loss_util import LossUtil
from utils.optimizer_util import OptimizerUtil
from pathlib import Path
from utils.optimizer_util import AbstractOptimizer

from torch.utils.tensorboard import SummaryWriter


class Config:
    """
    配置类：配置各种路径等配置
    """

    def __init__(self, train_data_path, test_data_path, log_dir, weight_path, load_weight, folder_path,
                 loss_function: LossUtil,
                 optimizer: AbstractOptimizer):
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
        self.folder_path = folder_path

        now = datetime.datetime.now()
        now = str(now).replace(':', '.')
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, str(len(os.listdir(log_dir)) + 1) + '.' + str(now)))

        self.weight_path = weight_path
        self.weight_path_latest = os.path.join(self.weight_path, '\latest\latest_weight.pth')
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

    def recordOtherConfig(self, epoch, batch_size):
        self.epoch = epoch
        self.batch_size = batch_size

    def recordTime(self,start_time,end_time):
        self.start_time = start_time
        self.end_time = end_time

    def recordResult(self):
        pass

def get_config():
    train_data_path = r'D:\Dataset\dataset_1000\train'
    test_data_path = r'D:\Dataset\dataset_1000\test'
    log_dir = r'log'
    weight_path = r'weight\weight_latest.pth'
    folderPath = r'trainFolder'
    load_weight = 0
    # loss_function = LossUtil.FocalLoss(torch.tensor([1, 1, 1]), 2)
    loss_function = LossUtil.CrossEntropyLoss(255)
    optimizer = OptimizerUtil.SGD(0.001, 0.8, 1e-2)

    return Config(train_data_path, test_data_path, log_dir, weight_path, load_weight, folderPath, loss_function,
                  optimizer)


"""
在folderPath下创建YYYYMMDD-x的文件夹并且返回路径
"""


def createAndGetFolder(folder_path):
    folders = [item.name for item in Path(folder_path).iterdir() if item.is_dir()]
    folders.sort(reverse=True)

    currentDatetime = datetime.datetime.now()
    currentDateStr = currentDatetime.strftime("%Y%m%d")

    if len(folders) == 0:
        newFolderName = currentDateStr + '-1'
    else:
        lastFolderName = folders[0]
        lastFolderDateStr = lastFolderName[0:8]
        newFolderName = lastFolderDateStr
        if (int(lastFolderDateStr) > int(currentDateStr)):
            raise Exception("日期错误")
        elif (int(lastFolderDateStr) == int(currentDateStr)):
            newFolderName += '-' + str(int(lastFolderName[lastFolderName.rfind('-') + 1:]) + 1)
        else:
            newFolderName += '-1'

    newFolderPath = os.path.join(folder_path, newFolderName)
    os.mkdir(newFolderPath)
    return newFolderPath

def recordConfig(config:Config):
    with open(os.path.join(config.folder_path,'config.txt'),'w') as f:
        trainDatesetSize = len(os.listdir(config.train_data_path))
        testDatasetSize = len(os.listdir(config.test_data_path))
        f.write(f"trainDatasetSize:{trainDatesetSize}\n")
        f.write(f"testDatasetSize:{testDatasetSize}\n")
        f.write(f"oneEpochUsePicCount:{trainDatesetSize + testDatasetSize}\n")
        f.write("--------------------------------------------------------------\n")
        f.write(f"batchSize:{config.batch_size}\n")
        f.write(f"epoch:{config.epoch}\n")
        f.write(f"optimizer:{config.optimizer.get_optimizer().__class__.__name__}\n")
        f.write(f"optimizerConfig:{config.optimizer.configtoStr()}\n")
        f.write("--------------------------------------------------------------\n")
        f.write(f"costTime:{config.end_time - config.start_time}\n")

def printConfig(config:Config):
    pass



if __name__ == '__main__':
    print(createAndGetFolder(get_config().folder_path))
