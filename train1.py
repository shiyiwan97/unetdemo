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
    # py的三元运算符
    print('cuda' if torch.cuda.is_available() else 'cpu')
    # trainDataPath = r'D:\Dataset\dataset_1000\train'
    # testDataPath = r'D:\Dataset\dataset_1000\test'
    trainDataPath = r'D:\Dataset\dataset_cpu_1'
    testDataPath = r'D:\Dataset\dataset_cpu_1'
    log_dir = 'log'
    now = datetime.datetime.now()
    now = str(now).replace(':', '.')

    #tensorboard的SummaryWriter，将数据写入log_dir中的事件文件以供tensorboard使用
    writer = SummaryWriter(log_dir=os.path.join(log_dir, str(len(os.listdir(log_dir)) + 1) + '.' + now))
    epoch_num = 200

    trainDataset = MyDataset(trainDataPath)
    testDataset = MyDataset(testDataPath)
    trainData = DataLoader(trainDataset, batch_size=9, shuffle=True)
    testData = DataLoader(testDataset, batch_size=9, shuffle=True)

    weightPathLatest = 'weight/latest/weight_latest.pth'
    weightPathIoU = 'weight/maxIoU/weight_max_iou.pth'
    IoURecordPath = 'weight/maxIoU/iou'
    weightPathLoss = 'weight/minLoss/weight_min_loss.pth'
    LossRecordPath = 'weight/minLoss/loss'

    # 数据和模型都默认在cpu上，如果device在是cpu，那.to(device)前后都是在cpu上，如果device是gpu，.to(device)会把数据或模型放到cpu上去
    # 可用通过.device去看是在cpu还是在gpu
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

    # loss = nn.CrossEntropyLoss()
    # 因为这个函数以第二个维度为类别维度，所以标签的值在0到第二个维度的大小-1之间，但是这里的掩码图用255标识背景，所以需要忽略255，否则会报错
    loss = nn.CrossEntropyLoss(ignore_index=255)

    #带动量的随机梯度下降Stochastic Gradient Descent
    #学习率=0.01
    #动量=0.8。普通SGD，参数更新=学习率*梯度，带动量的SGD，参数更新=动量*上一次参数更新+学习率*梯度。可以加速平缓区域的训练，一般设置为0.9
    #权重衰减=1e-2，权重衰减是为了防止过拟合，一般设置为1e-4到1之间
    opt = optim.SGD(unet.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-2)

    epoch = 1
    while epoch < epoch_num:
        trainLossSum = 0
        # 批次的数量，而不是图像的数量，数值=图像数量/批次大小向上取整
        trainPicCount = len(trainDataset)
        for i, (image, segment_image) in enumerate(tqdm.tqdm(trainData)):
            torch.cuda.empty_cache()
            image, segment_image = image.to(device), segment_image.to(device)
            out_image = unet(image)
            # loss的使用：默认第二个维度是类别维度
            # (N,C,*)和(N,*)，前者是预测值，后者是真实值，第二个维度是类别维度，后面可以无限扩展
            # 预测值直接使用网络的原始输出，因为内部包含了softmax函数
            train_loss = loss(
                # F.softmax(out_image.to(device)),
                out_image.to(device), #(1,19,512,512)
                # F.one_hot(np.squeeze(segment_image, axis=1).long(),19).permute(0, 3, 2, 1).float()
                segment_image.long() #(1,512,512)
            )
            # 优化器每次使用时要梯度归零
            # 其中梯度储存在每个参数的.grad属性中，参数=权重和偏置
            # model.parameters()返回的是一个生成器（迭代器），可以遍历，如果想要用下标访问可以list(model.parameters())转化为列表
            # list(model.parameters())[0].grad就是第一个参数的梯度
            # 更新梯度就是把新的梯度加到原来的梯度上，所以要先清零
            opt.zero_grad()
            # 反向传播梯度，就是把梯度传递给每个参数，存放在.grad属性中
            train_loss.backward()
            # 根据当前的梯度和优化测了去更新参数
            opt.step()
            # .item()是单个张量获取数值的方法
            trainLossSum += train_loss.item()
        torch.save(unet.state_dict(), weightPathLatest)
        trainLossTotal = trainLossSum / trainPicCount

        # f：模板语法，格式化字符串。
        print(f'epoch:{epoch}-batchNo:{i}-train_loss===>>{trainLossTotal}')

        # iouSum = iouCount = 0
        for i, (image, segment_image) in enumerate(tqdm.tqdm(testData)):
            testLossSum = 0
            testPicCount = len(testDataset)
            image, segment_image = image.to(device), segment_image.to(device)
            outImage = unet(image)
            predImage = outImage.permute(0, 2, 3, 1)
            predImage = CommonUtil.convertToL(predImage)
            predImage = torch.tensor(predImage).permute(0, 3, 1, 2)
            # predImage = predImage.squeeze(0)
            iou = CommonUtil.calculateIoU(predImage.squeeze(0), segment_image, 19)[1]
            iou2 = EvaluationUtil.calculate_IoU(predImage, segment_image)
            # iouSum += iou
            # iouCount += 1
            testLoss = loss(outImage.to(device), segment_image.long())
            testLossSum += testLoss
        testLossTotal = testLossSum / testPicCount
        # iouTotal = iouSum / iouCount
        # print(f'{epoch}-{i}-test_loss===>>{testLossTotal}    iou===>>{iouTotal}')
        print(f'epoch:{epoch}-batchNo:{i}-test_loss===>>{testLossTotal}')

        # 保存IoU最大的模型以及记录IoU数据
        # writer.add_scalar('iou', iouTotal, epoch)
        # iouFileR = open(IoURecordPath, 'r+')
        # line = iouFileR.readline().replace('\n', '')
        # splitArray = line.split(' ')
        # maxIoU = splitArray[3]
        # maxIoUEpoch = splitArray[2]
        # epoch = int(splitArray[0]) + 1
        #
        # iouFileR.seek(0)
        # if iouTotal > float(maxIoU):
        #     iouFileR.write(str(epoch) + ' max ' + str(epoch) + ' ' + str(iouTotal.item()) + '\n')
        #     torch.save(unet.state_dict(), weightPathIoU)
        # else:
        #     iouFileR.write(str(epoch) + ' max ' + maxIoUEpoch + ' ' + maxIoU + '\n')
        # iouFileR.close()
        # iouFileW = open(IoURecordPath, 'a')
        # iouFileW.write(str(epoch) + ' ' + str(iouTotal.item()) + '\n')
        # iouFileW.close()

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
