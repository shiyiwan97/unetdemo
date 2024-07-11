import torch
from torch.utils.data import DataLoader

from net.net import *
import os
from utils.common_util import CommonUtil
from utils.loss_util import LossUtil
from utils.optimizer_util import OptimizerUtil
from utils.evaluation_util import EvaluationUtil
from utils.save_weight_util import SaveWeightUtil
from data import *
from config import Config
import tqdm
import datetime
from tqdm import tqdm
import logging

from torch.utils.tensorboard import SummaryWriter



class NewTrain:

    @classmethod
    def train(self, config: Config):

        # 1.判断设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 2.初始化网络
        unet = UNet().to(device)
        # 3.读取权重
        CommonUtil.load_weight(unet, r'.\weight\baseline\weight_2.pth', device)
        # 4.定义损失函数
        loss_function = LossUtil.CrossEntropyLoss(255).get_loss()
        # 5.定义优化器
        optimizer = config.optimizer.get_optimizer(unet.parameters())
        # 6.数据集
        train_dataset = MyDataset(config.train_data_path)
        test_dataset = MyDataset(config.test_data_path)
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

        # 7. tensorboard
        config.setStartTime(datetime.datetime.now())
        trainWriter = config.writers['train']
        testWriter = config.writers['test']
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        logger = logging.getLogger()
        # 8.训练&评价
        config.generateDesc()
        logger.info(config.desc)
        epoch = 1
        startEpochTime = datetime.datetime.now()
        while epoch <= config.epoch:
            torch.cuda.empty_cache()
            train_loss_sum = 0
            train_pic_count = len(train_dataset)
            test_pic_count = len(test_dataset)
            mIoU_sum = 0.
            test_loss_sum = 0.

            for i, (image, segment_image) in enumerate(tqdm(train_dataloader)):
                image = image.to(device)
                segment_image = segment_image.to(device)
                out_image = unet(image)
                loss = loss_function(out_image, segment_image.long())
                optimizer.zero_grad()
                loss.backward()
                # 梯度会累计，每一次更新参数的时候需要归零；更新参数也依赖于梯度，所以先要反向传播计算梯度
                optimizer.step()
                # loss.item()只是获取损失的副本，一般不会受其他操作的影响
                train_loss_sum += loss.item()
            torch.save(unet.state_dict(), config.weightPath + r'\weight_' + str(epoch) + r'.pth')
            trainLossMean = train_loss_sum / train_pic_count
            logger.info(f'epoch:{epoch}   train_loss_mean:{trainLossMean}')

            with torch.no_grad():
                for i, (image, segment_image) in enumerate(tqdm(test_dataloader)):
                    image = image.to(device)
                    segment_image = segment_image.to(device)
                    out_image = unet(image)
                    loss = loss_function(out_image, segment_image.long()).item()
                    mIoU = EvaluationUtil.calculate_mIoU(out_image, segment_image, 19, [255], device).item()
                    mIoU_sum += mIoU
                    test_loss_sum += loss

            # 8.更新最佳权重
            best_value_type_max = 1
            best_value_type_min = 0
            mean_mIoU = mIoU_sum / test_pic_count
            testLossMean = test_loss_sum / test_pic_count
            # SaveWeightUtil.save_weight('mIoU', mean_mIoU, best_value_type_max, r'.\weight', unet.state_dict())
            # SaveWeightUtil.save_weight('loss', testLossMean, best_value_type_min, r'.\weight', unet.state_dict())

            # writer.add_scalar('loss/test-loss', testLossMean, epoch)
            trainWriter.add_scalar('loss', trainLossMean, epoch)
            # 因为每个epoch，每次dataloader读图时都会更新梯度，所以测试集的损失对应下一次训练集的损失比较好
            testWriter.add_scalar('loss', testLossMean, epoch + 1)
            epoch += 1

        endEpochTime = datetime.datetime.now()
        config.setEndTime(endEpochTime)
        # config.setResult()
        config.recordConfig(config.trainFolder)
        config.writeConfigToLog()
        config.closeWriter()


if __name__ == '__main__':
    config = Config.get_config()
    NewTrain.train(config)
