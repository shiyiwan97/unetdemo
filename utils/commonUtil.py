import os
import time

from PIL import Image
import numpy as np
import cv2
from multiprocessing import Process
import shutil
import torch

raw_path = 'F:\machineLearning\dataset\dataset_1000'
save_path = 'F:\machineLearning\dataset\dataset_1000'


class CommonUtil:

    # 获取特定数据集（原图、seg、ldmks混在一起的）的seg图uriList
    @classmethod
    def getUriList(self, path, match):

        fileList = os.listdir(path)
        returnList = []
        for i in range(len(fileList)):
            if (fileList[i].__contains__(match)):
                returnList.append(fileList[i])
        return returnList

    # 多线程划分list(processCount = 进程数，processNum=进程编号，从0开始)
    @classmethod
    def divideList(self, list, processCount, processNo):

        returnList = []

        for i in range(len(list)):
            if (i % processCount == processNo):
                returnList.append(list[i])

        if (processNo == 0):
            returnList.__add__(list[:-len(list) % processCount])

        return returnList

    # 缩放一张图片，只支持整数倍缩小，灰度图
    @classmethod
    def shrinkPicL(self, rawUri, shrinkUri, widthIn, heightIn, widthOut, heightOut):
        # time1 = time.time()
        if (heightIn % heightOut != 0 or widthIn % widthOut != 0):
            raise Exception("只支持整数倍缩小！")

        heightFactor = heightIn / heightOut
        widthFactor = widthIn / widthOut

        dataArray = [[0 for i in range(widthOut)] for j in range(heightOut)]
        rawArray = np.array(Image.open(rawUri))

        for i in range(heightOut):
            for j in range(widthOut):
                dataArray[i][j] = self.calculateShrinkPixelValue(widthFactor, heightFactor, i, j, rawArray, 0)
        out = Image.fromarray(np.array(dataArray), "L")
        # time2 = time.time()
        out.save(shrinkUri)
        # time3 = time.time()
        # print(str(os.getpid()) + '  ==> cpu: ' + str(time2 - time1) + ' io: ' + str(time3 - time2))
        # out.show()

    # 计算缩放后某个位置的值
    @classmethod
    def calculateShrinkPixelValue(self, widthFactor, heightFactor, x, y, rawArray, add):

        # add表示左右/上下扩展的像素个数 todo 次数问题
        if add > 100:
            raise Exception("扩展了100次还是有多个出现次数最多像素值！")

        # todo 无法向边界扩展怎么办？
        # 获取缩小后图片(x,y)处的像素对应原图的像素范围
        height = len(rawArray)
        width = len(rawArray[0])
        rawPicXStart = (int)(x * widthFactor - add if x * widthFactor - add >= 0 else 0)
        rawPicXEnd = (int)((x + 1) * widthFactor + add - 1 if (x + 1) * widthFactor + add - 1 <= width else width)
        rawPicYStart = (int)(y * heightFactor - add if y * heightFactor - add >= 0 else 0)
        rawPicYEnd = (int)((y + 1) * heightFactor + add - 1 if (y + 1) * heightFactor + add - 1 <= height else height)

        # 截取对应原图区域的像素值二维数组并展开为一维
        shrinkArray = rawArray[rawPicXStart:rawPicXEnd + 1, rawPicYStart:rawPicYEnd + 1].ravel()

        # 放入字典，key=数字，value=出现次数
        countDictionary = dict()
        for i in range(len(shrinkArray)):
            countDictionary[shrinkArray[i]] = \
                countDictionary[shrinkArray[i]] + 1 if shrinkArray[i] in countDictionary.keys() else 1

        # 获取字典values，并排序，如果最高次数数字有多个，则add+1调用自身
        countValues = countDictionary.values()
        countArray = (list)(countValues)
        sortedCountArray = sorted(countValues)
        if (len(sortedCountArray) > 1 and sortedCountArray[-1] == sortedCountArray[-2]):
            self.calculateShrinkPixelValue(widthFactor, heightFactor, x, y, rawArray, add + 1)

        return ((list)(countDictionary.keys()))[countArray.index(max(countArray))]

    @classmethod
    def shrinkPicsFromDir(self, rawDir, shrinkDir, widthIn, heightIn, widthOut, heightOut, processCount, processNo):
        rawPicNameListAll = CommonUtil.getUriList(rawDir, 'png')
        rawPicNameList = CommonUtil.divideList(rawPicNameListAll, processCount, processNo)
        for i in range(len(rawPicNameList)):
            if (not os.path.exists(os.path.join(shrinkDir, rawPicNameList[i]))):
                if (rawPicNameList[i].__contains__('_seg.png')):
                    CommonUtil.shrinkPicL(os.path.join(rawDir, rawPicNameList[i]),
                                          os.path.join(shrinkDir, rawPicNameList[i]), widthIn, heightIn, widthOut,
                                          heightOut)
                else:
                    img = cv2.imread(os.path.join(rawDir, rawPicNameList[i]))
                    resizeImg = cv2.resize(img, (widthOut, heightOut))
                    cv2.imwrite(os.path.join(shrinkDir, rawPicNameList[i]), resizeImg)

    # np.set_printoptions(threshold=np.inf) 查看矩阵全部值

    @classmethod
    def testProcessCount(self, maxCount):
        countTimeDic = dict()

        for i in range(1, maxCount + 1):
            time1 = time.time()
            for j in range(i):
                process = Process(target=CommonUtil.shrinkPicsFromDir, args=('F:\machineLearning\dataset\dataset_100',
                                                                             'F:\machineLearning\dataset\dataset_100_shrink',
                                                                             512, 512, 256, 256, i, j))
                process.start()
            while (True):
                if (len(os.listdir('F:\machineLearning\dataset\dataset_100_shrink')) == 200):
                    time2 = time.time()
                    countTimeDic[i] = time2 - time1
                    print(str(i) + '  ' + str(time2 - time1))
                    shutil.rmtree('F:\machineLearning\dataset\dataset_100_shrink')
                    os.mkdir('F:\machineLearning\dataset\dataset_100_shrink')
                    break

        print(countTimeDic)

    @classmethod
    def calculateIoU(self, pred, mask, classNum):
        """
        计算iou
        :param pred:预测得到的tensor
        :param mask: 标签tnesor
        :return: ioutensor
        """
        if not pred.size() == mask.size():
            raise Exception('维数不一致，无法计算IOU')

        iou = []
        pred = pred.view(-1)
        mask = mask.view(-1)

        for classIndex in range(classNum):
            predIndex = pred == classIndex
            maskIndex = mask == classIndex

            intersection = predIndex[maskIndex].sum()
            intersection2 = maskIndex[predIndex].sum()
            union = predIndex.sum() + maskIndex.sum() - intersection
            iou.append(intersection.float() / union.float())

            global sum, count
            sum = count = 0

        for i in range(len(iou)):
            if not torch.isnan(iou[i]):
                sum += iou[i]
                count += 1
        return iou, 0 if count == 0 else sum / count

    @classmethod
    def convertToL(self, tensor):
        """
        转换为灰度图
        :param tensor:tensor（c,h,w） c=1
        :return: tensor(c,h,w) c=1
        """
        width = len(tensor[0])
        height = len(tensor)
        result = torch.empty(1, width, height)

        for i in range(width):
            for j in range(height):
                result[0, i, j] = torch.max(tensor[i, j], dim=0)[1]

        nparray = np.array(result.permute(1, 2, 0))
        return nparray


if __name__ == '__main__':
    import torch

    t1 = torch.tensor([1, 2, 1])
    t2 = torch.tensor([1, 3, 1])

    iou1 = CommonUtil.calculateIoU(t1, t2, 6)
    print(iou1)

    # test = 1
    # CommonUtil.testProcessCount(20)

    # p = Process(target=print,args='done')
    # p.start()
    # p.join()

    # startTime = time.time()
    # CommonUtil.shrinkPicsFromDir('F:\machineLearning\dataset\dataset_100',
    #                              'F:\machineLearning\dataset\dataset_100_shrink', 512, 512, 256, 256)
    # endTime = time.time()
    # print('total time: ' + str(endTime - startTime))

    #
    # shrinkArray = [1, 2, 3, 4, 4, 5, 2, 4, 5]
    # countDictionary = dict()
    # for i in range(len(shrinkArray)):
    #     int = shrinkArray[i]
    #     countDictionary[int] = countDictionary[int] + 1 if int in countDictionary.keys() else 1
    # a = list(countDictionary.values())
    # a.sort()
    # print(a)
    # print(np.bincount([1, 2, 2, 3, 4, 4]))
    # print(np.argmax(np.bincount([1, 2, 2, 3, 4, 4])))
    # print(np.argmax(1))
    # print(countDictionary.values())

    # image1 = Image.open("F:\machineLearning\dataset\dataset_1000\\000000.png")
    # print(image1.getpixel((0, 0)))
    # image2 = Image.open("F:\machineLearning\dataset\dataset_1000\\000000_seg.png")
    # print(image2.getpixel((0, 0)))
    #
    # print(len(np.array(image2)))
    # print(len(np.array(image2)[0]))
    #
    # print("-----------------------------")
    # array = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
    # array1 = np.array(array)
    # print(array1[0:2, 0:2].ravel())
    #
    # list = ['1', '2', '3', '6', '5','5', '6', '6', '2', '1','5']
    # result = max(set(list), key=list.count)
