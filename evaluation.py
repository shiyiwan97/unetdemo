import os

import torch
from net.net import UNet
from utils.common_util import CommonUtil
from utils.evaluation_util import EvaluationUtil
from torch.utils.data import DataLoader
from data import *
from utils.loss_util import LossUtil
import tqdm


class Evaluation:

    @classmethod
    def evaluation(cls, weight_folder_uri, dataset_folder_uri, batch_size, record_folder_uri):
        # 1.判断设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 2.初始化网络
        unet = UNet().to(device)
        # 3.数据集
        test_dataset = MyDataset(dataset_folder_uri)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        test_data_size = len(test_dataset)
        # 4.定义损失函数
        loss = LossUtil.CrossEntropyLoss(255).get_loss()
        # 5.获取权重文件uri
        weight_uris = CommonUtil.get_uri_from_folder(weight_folder_uri, 'pth')

        # 6.打开记录文件输出
        loss_file_a = open(os.path.join(record_folder_uri, 'loss'), 'a')
        mIoU_file_w = open(os.path.join(record_folder_uri, 'mIoU'), 'a')
        iou_file_a = open(os.path.join(record_folder_uri, 'IoU'), 'a')

        for i in range(len(weight_uris)):
            # 3.读取权重
            CommonUtil.load_weight(unet, weight_uris[i], device)
            with torch.no_grad():
                loss_sum = 0.
                mIoU_sum = 0.
                iou_mean = torch.zeros([19])
                for j, (image, segment_image) in enumerate(tqdm.tqdm(test_dataloader)):
                    image = image.to(device)
                    segment_image = segment_image.to(device)
                    out_image = unet(image)
                    loss_sum += loss(out_image, segment_image.long())

                    segmentation_metric = EvaluationUtil.get_segmentation_metric(out_image, segment_image, 19, [255],
                                                                                 device)
                    mIoU_sum += segmentation_metric.meanIntersectionOverUnion().item()
                    iou = segmentation_metric.IntersectionOverUnion()
                loss_mean = loss_sum / test_data_size
                mIoU_mean = mIoU_sum / test_data_size
                iou_mean += iou / test_data_size

            loss_file_a.write(weight_uris[i] + ' ' + str(loss_mean) + '\n')
            mIoU_file_w.write(weight_uris[i] + ' ' + str(mIoU_mean) + '\n')
            iou_file_a.write(weight_uris[i] + ' ' + str(iou_mean) + '\n')

        loss_file_a.close()
        mIoU_file_w.close()
        iou_file_a.close()

    @classmethod
    def find_best(cls, file_info):
        for i in range(len(file_info)):
            file_uri = file_info[i][0]
            best_value_type = file_info[i][1]
            value_index = file_info[i][2]
            values = []

            with open(file_uri, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    values.append(line.split(' ')[value_index])

            if best_value_type == 'max':
                best_value = max(values)
            else:
                best_value = min(values)

            best_value_index = values.index(best_value)
            print(file_uri + ' ' + str(best_value_index) + ' ' + str(best_value))


if __name__ == '__main__':
    Evaluation.find_best([[r'evaluation\record\mIoU', 'max', 1], [r'evaluation\record\loss', 'min', 1]])

    # Evaluation.evaluation(r'F:\machineLearning\unetdemo\weight\latest', r'F:\machineLearning\dataset\test\2pics', 2,
    #                       r'evaluation\record')
