import datetime
import os
from utils.loss_util import LossUtil
from pathlib import Path
from utils.optimizer_util import OptimizerUtil

from torch.utils.tensorboard import SummaryWriter


class Config:
    """
    配置类：配置各种路径等配置
    """

    def __init__(self, train_data_path, test_data_path, load_weight, folder_path, epoch,
                 batch_size, loss_function: LossUtil, optimizer: OptimizerUtil.AbstractOptimizer):
        """

        :param train_data_path:
        :param test_data_path:
        :param log_dir:
        :param weight_path:
        :param load_weight: 0=latest,1=max_iou,3=min_loss
        :param loss_function:
        """
        self.folder_path = folder_path
        self.createAndGetFolder()
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.batch_size = batch_size
        self.epoch = epoch

        now = datetime.datetime.now()
        now = str(now).replace(':', '.')
        train_writer = SummaryWriter(log_dir=os.path.join(self.logDir, 'train'))
        test_writer = SummaryWriter(log_dir=os.path.join(self.logDir, 'test'))
        self.writers = {'train': train_writer, 'test': test_writer}
        self.weight_path_latest = os.path.join(self.weightPath, '\latest\latest_weight.pth')
        self.loss_record_path = os.path.join(self.weightPath, 'iou')
        self.weight_path_iou = os.path.join(self.weightPath, 'max_IoU_weight.pth')
        self.iou_record_loss = os.path.join(self.weightPath, 'loss')
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

    def setStartTime(self,start_time):
        self.start_time = start_time

    def setEndTime(self,end_time):
        self.end_time = end_time

    def setResult(self, min_loss):
        self.min_loss = min_loss

    @staticmethod
    def get_config():
        train_data_path = r'D:\Dataset\dataset_1'
        test_data_path = r'D:\Dataset\dataset_1'
        epoch = 5
        batch_size = 8
        folderPath = r'trainFolder'
        load_weight = 0
        # loss_function = LossUtil.FocalLoss(torch.tensor([1, 1, 1]), 2)
        loss_function = LossUtil.CrossEntropyLoss(255)
        optimizer = OptimizerUtil.SGD(0.001, 0.8, 1e-2)

        return Config(train_data_path, test_data_path, load_weight, folderPath, epoch, batch_size,
                      loss_function, optimizer)

    """
    在folderPath下创建YYYYMMDD-x的文件夹并且返回路径
    """

    def createAndGetFolder(self):
        folders = [item.name for item in Path(self.folder_path).iterdir() if item.is_dir()]

        currentDatetime = datetime.datetime.now()
        currentDateStr = currentDatetime.strftime("%Y%m%d")

        if len(folders) == 0:
            newFolderName = currentDateStr + '-1'
        else:
            folders = sorted(folders,key=lambda x:(int(x.split('-')[0]), int(x.split('-')[1])),reverse=True)
            lastFolderName = folders[0]
            lastFolderDateStr = lastFolderName[0:8]
            newFolderName = lastFolderDateStr
            if (int(lastFolderDateStr) > int(currentDateStr)):
                raise Exception("日期错误")
            elif (int(lastFolderDateStr) == int(currentDateStr)):
                newFolderName += '-' + str(int(lastFolderName[lastFolderName.rfind('-') + 1:]) + 1)
            else:
                newFolderName = currentDateStr + '-1'

        self.trainFolder = os.path.join(self.folder_path, newFolderName)
        os.mkdir(self.trainFolder)
        self.weightPath = os.path.join(self.trainFolder, 'weight')
        self.logDir = os.path.join(self.trainFolder, 'log')
        os.mkdir(self.weightPath)
        os.mkdir(self.logDir)
        return self.trainFolder


    def recordConfig(self, recordPath):
        with open(os.path.join(recordPath, 'config.txt'), 'w') as f:
            f.write(self.desc)

    def printConfig(self):
        pass

    def writeConfigToLog(self):
        self.writers['train'].add_text('train',self.desc)

    def closeWriter(self):
        for writer in self.writers.values():
            writer.close()

    def generateDesc(self):
        trainDatesetSize = int(len(os.listdir(self.train_data_path)) / 3)
        testDatasetSize = int(len(os.listdir(self.test_data_path)) / 3)
        self.desc = f"""
        trainDatasetSize: {trainDatesetSize}
        testDatasetSize: {testDatasetSize}
        oneEpochUsePicCount: {trainDatesetSize + testDatasetSize}
        --------------------------------------------------------------
        batchSize: {self.batch_size}
        epoch: {self.epoch}
        optimizer: {self.optimizer.__class__.__name__}
        optimizerConfig: {self.optimizer.configtoStr()}
        --------------------------------------------------------------
        """
        if (hasattr(self,'end_time')):
            self.desc += f"""costTime: {self.end_time - self.start_time}"""

if __name__ == '__main__':
    print(Config.get_config().createAndGetFolder())
