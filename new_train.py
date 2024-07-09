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
        test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

        # 7. tensorboard
        config.setStartTime(datetime.datetime.now())
        writer = config.writer
        # 8.训练&评价
        epoch = 1
        startEpochTime = datetime.datetime.now()
        while epoch <= config.epoch:
            torch.cuda.empty_cache()
            train_loss_sum = 0
            train_pic_count = len(train_dataset)
            test_pic_count = len(test_dataset)
            mIoU_sum = 0.
            loss_sum = 0.

            for i, (image, segment_image) in enumerate(tqdm.tqdm(train_dataloader)):
                image = image.to(device)
                segment_image = segment_image.to(device)
                out_image = unet(image)
                loss = loss_function(out_image, segment_image.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item()
            torch.save(unet.state_dict(), config.weightPath + r'\weight_' + str(epoch) + r'.pth')
            trainLossTotal = train_loss_sum / train_pic_count
            print(f'{epoch}-train_loss_FL===>>{trainLossTotal}')

            writer.add_scalar('loss', trainLossTotal, epoch)

            with torch.no_grad():
                for i, (image, segment_image) in enumerate(tqdm.tqdm(test_dataloader)):
                    image = image.to(device)
                    segment_image = segment_image.to(device)
                    out_image = unet(image)
                    loss = loss_function(out_image, segment_image.long()).item()
                    mIoU = EvaluationUtil.calculate_mIoU(out_image, segment_image, 19, [255], device).item()
                    mIoU_sum += mIoU
                    loss_sum += loss

            # 8.更新最佳权重
            best_value_type_max = 1
            best_value_type_min = 0
            mean_mIoU = mIoU_sum / test_pic_count
            # mean_loss = loss_sum / test_pic_count
            # SaveWeightUtil.save_weight('mIoU', mean_mIoU, best_value_type_max, r'.\weight', unet.state_dict())
            # SaveWeightUtil.save_weight('loss', mean_loss, best_value_type_min, r'.\weight', unet.state_dict())

            epoch += 1

        endEpochTime = datetime.datetime.now()
        config.setEndTime(endEpochTime)
        # config.setResult()
        config.recordConfig(config.trainFolder)
        writer.close()


if __name__ == '__main__':
    config = Config.get_config()
    NewTrain.train(config)
