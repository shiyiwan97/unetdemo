import os

from PIL import Image
import numpy as np

raw_path = 'F:\machineLearning\dataset\dataset_1000'


class ChangeSizeUtil:

    # 获取特定数据集（原图、seg、ldmks混在一起的）的seg图uriList
    def getUriList(self, path):

        fileList = os.listdir(path)
        returnList = []
        for i in range(len(fileList)):
            if (fileList[i].__contains__("_seg.png")):
                returnList.add(fileList[i])
        return returnList

    # 缩放一张图片，只支持整数倍缩小
    @classmethod
    def shrinkPic(self, uri, widthIn, heightIn, widthOut, heightOut):

        if (heightIn % heightOut != 0 or widthIn % widthOut != 0):
            raise Exception("只支持整数倍缩小！")

        heightFactor = heightIn / heightOut
        widthFactor = widthIn / widthOut

        dataArray = [[0 for i in range(widthOut)] for j in range(heightOut)]
        rawArray = np.array(Image.open(uri))

        for i in range(heightOut):
            for j in range(widthOut):
                dataArray[i][j] = self.calculateShrinkPixelValue(widthFactor, heightFactor, i, j, rawArray, 0)
        out = Image.fromarray(np.array(dataArray), "L")
        out.save("F:\machineLearning\dataset\\test\\000000_seg——resize.png")
        out.show()

    # 计算缩放后某个位置的值
    @classmethod
    def calculateShrinkPixelValue(self, widthFactor, heightFactor, x, y, rawArray, add):

        # add表示左右/上下扩展的像素个数 todo 次数问题
        if add > 20:
            raise Exception("扩展了5次还是有多个出现次数最多像素值！")

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

# np.set_printoptions(threshold=np.inf) 查看矩阵全部值
if __name__ == '__main__':
    util = ChangeSizeUtil()
    ChangeSizeUtil.shrinkPic("F:\machineLearning\dataset\\test\\000000_seg.png", 512, 512, 256, 256)

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
