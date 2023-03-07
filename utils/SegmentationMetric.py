import torch
import numpy as np
class SegmentationMetric(object):
    def __init__(self, numClass,device):
        self.numClass = numClass
        self.confusionMatrix = torch.zeros((self.numClass,) * 2)  # 混淆矩阵（空）
        self.confusionMatrix = self.confusionMatrix.to(device)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy 正确的像素占总像素的比例
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = torch.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        """
        Mean Pixel Accuracy(MPA，均像素精度)：是PA的一种简单提升，计算每个类内被正确分类像素数的比例，之后求所有类的平均。
        :return:
        """
        classAcc = self.classPixelAccuracy()
        meanAcc = classAcc[classAcc < float('inf')].mean() # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = torch.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = torch.sum(self.confusionMatrix, axis=1) + torch.sum(self.confusionMatrix, axis=0) - torch.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        return IoU

    def meanIntersectionOverUnion(self):
        IoU = self.IntersectionOverUnion()
        mIoU = IoU[IoU<float('inf')].mean()# 求各类别IoU的平均
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel, ignore_labels):  #
        """
        同FCN中score.py的fast_hist()函数,计算混淆矩阵
        :param imgPredict:
        :param imgLabel:
        :return: 混淆矩阵
        """
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        if ignore_labels!=None:
            for IgLabel in ignore_labels:
                mask &= (imgLabel != IgLabel)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = torch.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.view(self.numClass, self.numClass)
        # print(confusionMatrix)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        """
        FWIoU，频权交并比:为MIoU的一种提升，这种方法根据每个类出现的频率为其设置权重。
        FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        """
        freq = torch.sum(self.confusionMatrix, axis=1) / torch.sum(self.confusionMatrix)
        freq = freq.cpu()
        iu = np.diag(self.confusionMatrix.cpu()) / (
                torch.sum(self.confusionMatrix.cpu(), axis=1) + torch.sum(self.confusionMatrix.cpu(), axis=0) -
                torch.diag(self.confusionMatrix.cpu()))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel, ignore_labels):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel, ignore_labels)  # 得到混淆矩阵
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = torch.zeros((self.numClass, self.numClass))

if __name__ == '__main__':
    imgsize = 512
    batchsize = 16
    classnum = 19
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    metric = SegmentationMetric(classnum, device)  # 19种分类，有几个分类就填几

    ModelOut = torch.rand(batchsize, classnum, imgsize, imgsize).to(device) #网络预测输出值 BATCHSIZE * CLASSNUM * H * W
    imgLabel = torch.randint(0, classnum-1, (batchsize,imgsize, imgsize)).to(device) #标签  有19类所以取值范围 0 - 18

    imgPredictsoftmax = torch.softmax(input=ModelOut, dim=1)
    imgPredict = torch.argmax(imgPredictsoftmax, dim=1) #网络预测输出值转为预测图像分类值 和 imgLabel 尺度一致

    ignore_labels = [255] #如果没有需要忽略的写 ignore_labels = None
    hist = metric.addBatch(imgPredict, imgLabel, ignore_labels)  # 传入数据
    pa = metric.pixelAccuracy()
    cpa = metric.classPixelAccuracy()
    mpa = metric.meanPixelAccuracy()
    IoU = metric.IntersectionOverUnion()
    mIoU = metric.meanIntersectionOverUnion()
    fwIoU = metric.Frequency_Weighted_Intersection_over_Union()
    print("hist=", hist)  #混淆矩阵
    print("pa=",pa)
    print("cpa=", cpa)
    print("mpa=", mpa)
    print("IoU=", IoU)
    print("mIoU=", mIoU)
    print("fwIoU=", fwIoU)



