import torch
from torch.utils.data import DataLoader

from net.net import *
import os
from utils.common_util import CommonUtil
from utils.optimizer_util import OptimizerUtil
from utils.evaluation_util import EvaluationUtil
from utils.save_weight_util import SaveWeightUtil
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
        optimizer = config.optimizer.get_optimizer(unet.parameters())
        # 6.数据集
        train_dataset = MyDataset(config.train_data_path)
        test_dataset = MyDataset(config.test_data_path)
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)
        # 7.训练&评价
        epoch = 1
        while epoch < 200:
            train_loss_sum = 0
            train_pic_count = len(train_dataset)
            mIoU_sum = 0
            loss_sum = 0
            test_pic_count = len(test_dataset)

            for i, (image, segment_image) in enumerate(tqdm.tqdm(train_dataloader)):
                out_image = unet(image)
                loss = loss_function(out_image, segment_image.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item()
            torch.save(unet.state_dict(), config.weight_path_latest)
            trainLossTotal = train_loss_sum / train_pic_count
            print(f'{epoch}-{i}-train_loss===>>{trainLossTotal}')

            for i, (image, segment_image) in enumerate(tqdm.tqdm(test_dataloader)):
                out_image = unet(image)
                loss = loss_function(out_image, segment_image)
                mIoU = EvaluationUtil.calculate_mIoU(out_image, segment_image, 19, 255, device)
                mIoU_sum += mIoU
                loss_sum += loss

            # 8.更新最佳权重
            best_value_type_max = 1
            best_value_type_min = 0
            mean_mIoU = mIoU_sum / test_pic_count
            mean_loss = loss_sum / test_pic_count
            SaveWeightUtil.save_weight('mIoU', mean_mIoU, best_value_type_max, r'\weight', unet.state_dict())
            SaveWeightUtil.save_weight('loss', mean_loss, best_value_type_min, r'\weight', unet.state_dict())

        epoch += 1


if __name__ == '__main__':
    config = config_1.get_config()
    NewTrain.train(config)
