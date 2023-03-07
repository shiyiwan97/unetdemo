from sklearn.metrics import confusion_matrix
import numpy as np
from utils.SegmentationMetric import *


class EvaluationUtil:

    @classmethod
    def calculate_IoU(self, pred, label):
        matrix = self.calculate_confusion_matrix(pred, label)
        rowCount, colCount = matrix.shape

        if rowCount != colCount:
            raise Exception('混淆矩阵长宽不一致！')
        iouSum = 0
        for i in range(rowCount):
            TP = matrix[i][i]
            TN = np.sum(np.concatenate((matrix[i, 0:i], matrix[i, i + 1:rowCount + 1]), axis=0), axis=0)
            FP = np.sum(np.concatenate((matrix[0:i, i], matrix[i + 1:rowCount + 1, i]), axis=0), axis=0)
            iouSum += TP / (TP + TN + FP)

        return iouSum / rowCount

    @classmethod
    def calculate_confusion_matrix(self, pred, label):
        pred = pred.reshape(-1)
        label = label.reshape(-1)
        return confusion_matrix(label, pred)

    @classmethod
    def calculate_mIoU(self, pred, label, classnum, ignore, device):
        imgsize = pred.size(2)
        batchsize = pred.size(0)

        metric = SegmentationMetric(classnum, device)

        imgPredictsoftmax = torch.softmax(pred, dim=1)
        imgPredict = torch.argmax(imgPredictsoftmax, dim=1)

        ignore_labels = ignore
        hist = metric.addBatch(imgPredict, label, ignore_labels)

        mIoU = metric.meanIntersectionOverUnion()

        return mIoU
