import torch
from net.net import *
import os
from utils.common_util import CommonUtil
from utils.optimizer_util import OptimizerUtil
from data import *
import config_1
import tqdm


# from torch.utils.tensorboard import SummaryWriter


class NewTrain:

    @classmethod
    def train(self, config):

        # 1.判断设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 2.初始化网络
        unet = UNet().to(device)
        # 3.读取权重
        CommonUtil.load_weight(unet, config.load_weight_path, device)
        # 4.定义损失函数
        loss_function = config.loss_function.get_loss()
        # 5.定义优化器
        optimizer = config.optimizer.get_optimizer()
        # 6.数据集
        train_dataset = MyDataset(config.train_data_path)
        test_dataset = MyDataset(config.test_data_path)
        # 7.训练
        epoch = 1
        while epoch < 200:
            train_loss_sum = 0
            train_pic_count = len(train_dataset)
            for i, (image, segment_image) in enumerate(tqdm.tqdm(train_dataset)):
                out_image = unet(image)
                loss = loss_function(out_image, segment_image)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item()
            torch.save(unet.state_dict(), config.weight_path_latest)
            trainLossTotal = train_loss_sum / train_pic_count

            print(f'{epoch}-{i}-train_loss===>>{trainLossTotal}')


if __name__ == '__main__':
    config = config_1.get_config()
    NewTrain.train(config)
