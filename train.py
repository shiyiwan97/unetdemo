import os.path

from torch.utils.data import DataLoader
from data import *
import tqdm
from net.net import *
from torch import nn, optim
import torch.nn.functional as F
from utils.common_util import *
from torch.utils.tensorboard import SummaryWriter
import datetime
from utils.evaluation_util import EvaluationUtil


def Train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('cuda' if torch.cuda.is_available() else 'cpu')
    # trainDataPath = 'F:\machineLearning\dataset\dataset_1000'
    trainDataPath = r'F:\machineLearning\dataset\test\train'
    testDataPath = r'F:\machineLearning\dataset\test\test'
    log_dir = r'F:\machineLearning\dataset\test\log'
    now = datetime.datetime.now()
    now = str(now).replace(':', '.')

    writer = SummaryWriter(log_dir=os.path.join(log_dir, str(len(os.listdir(log_dir)) + 1) + '.' + now))
    epoch_num = 200

    trainDataset = MyDataset(trainDataPath)
    testDataset = MyDataset(testDataPath)
    trainData = DataLoader(trainDataset, batch_size=8, shuffle=True)
    testData = DataLoader(testDataset, batch_size=8, shuffle=True)

    weightPathLatest = 'weight/latest/weight_latest.pth'
    weightPathIoU = 'weight/maxIoU/weight_max_iou.pth'
    IoURecordPath = 'weight/maxIoU/iou'
    weightPathLoss = 'weight/minLoss/weight_min_loss.pth'
    LossRecordPath = 'weight/minLoss/loss'

    unet = UNet().to(device)
    if os.path.exists((weightPathLatest)):
        # unet = torch.load(weightPathLatest, map_location=torch.device('cpu'))
        if (device == torch.device('cpu')):
            unet.load_state_dict(torch.load(weightPathLatest, map_location=torch.device('cpu')))
        else:
            unet.load_state_dict(torch.load(weightPathLatest))
        print('读取权重数据成功！')
    else:
        print('读取权重数据失败！')

    loss = nn.CrossEntropyLoss(ignore_index=255)
    opt = optim.SGD(unet.parameters(), lr=0.01, momentum=0.8, weight_decay=1e-2)

    epoch = 1
    while epoch < 20000:
        trainLossSum = 0
        trainPicCount = len(trainDataset)
        for i, (image, segment_image) in enumerate(tqdm.tqdm(trainData)):
            torch.cuda.empty_cache()
            image, segment_image = image.to(device), segment_image.to(device)
            out_image = unet(image)

            train_loss = loss(
                # F.softmax(out_image.to(device)),
                out_image.to(device),
                # F.one_hot(np.squeeze(segment_image, axis=1).long(),19).permute(0, 3, 2, 1).float()
                segment_image.long()
            )
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            trainLossSum += train_loss.item()
        torch.save(unet.state_dict(), weightPathLatest)
        trainLossTotal = trainLossSum / trainPicCount

        print(f'{epoch}-{i}-train_loss===>>{trainLossTotal}')

        iouSum = iouCount = 0
        for i, (image, segment_image) in enumerate(tqdm.tqdm(testData)):
            testLossSum = 0
            testPicCount = len(testDataset)
            image, segment_image = image.to(device), segment_image.to(device)
            outImage = unet(image)
            predImage = outImage.permute(0, 2, 3, 1)
            predImage = CommonUtil.convertToL(predImage)
            predImage = torch.tensor(predImage).permute(0, 3, 1, 2)
            iou = CommonUtil.calculateIoU(predImage.squeeze(), segment_image, 19)[1]
            iou2 = EvaluationUtil.calculate_IoU(predImage, segment_image)
            iouSum += iou
            iouCount += 1
            testLoss = loss(outImage.to(device), segment_image.long())
            testLossSum += testLoss
        testLossTotal = testLossSum / testPicCount
        iouTotal = iouSum / iouCount
        print(f'{epoch}-{i}-test_loss===>>{testLossTotal}    iou===>>{iouTotal}')

        # 保存IoU最大的模型以及记录IoU数据
        writer.add_scalar('iou', iouTotal, epoch)
        iouFileR = open(IoURecordPath, 'r+')
        line = iouFileR.readline().replace('\n', '')
        splitArray = line.split(' ')
        maxIoU = splitArray[3]
        maxIoUEpoch = splitArray[2]
        epoch = int(splitArray[0]) + 1

        iouFileR.seek(0)
        if iouTotal > float(maxIoU):
            iouFileR.write(str(epoch) + ' max ' + str(epoch) + ' ' + str(iouTotal.item()) + '\n')
            torch.save(unet.state_dict(), weightPathIoU)
        else:
            iouFileR.write(str(epoch) + ' max ' + maxIoUEpoch + ' ' + maxIoU + '\n')
        iouFileR.close()
        iouFileW = open(IoURecordPath, 'a')
        iouFileW.write(str(epoch) + ' ' + str(iouTotal.item()) + '\n')
        iouFileW.close()
        epoch += 1
        # 保存Loss最小的模型以及记录Loss数据
        writer.add_scalar('loss', testLossTotal, epoch)
        lossFileR = open(LossRecordPath, 'r+')
        line = lossFileR.readline().replace('\n', '')
        splitArray = line.split(' ')
        minLoss = splitArray[3]
        minLossEpoch = splitArray[2]
        epoch = int(splitArray[0]) + 1

        lossFileR.seek(0)
        if testLossTotal < float(minLoss):
            lossFileR.write(str(epoch) + ' min ' + str(epoch) + ' ' + str(testLossTotal.item()) + '\n')
            torch.save(unet.state_dict(), weightPathLoss)
        else:
            lossFileR.write(str(epoch) + ' min ' + minLossEpoch + ' ' + minLoss + '\n')
        lossFileR.close()
        lossFileW = open(LossRecordPath, 'a')
        lossFileW.write(str(epoch) + ' ' + str(testLossTotal.item()) + '\n')
        lossFileW.close()
        epoch += 1
        writer.close()


if __name__ == '__main__':
    Train()
