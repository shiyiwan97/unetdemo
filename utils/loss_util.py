from abc import abstractmethod

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class LossUtil:
    class Loss:

        @abstractmethod
        def get_loss(self):
            pass

    class CrossEntropyLoss(Loss):

        def __init__(self, ignore):
            self.ignore = ignore

        def get_loss(self):
            return nn.CrossEntropyLoss(ignore_index=self.ignore)

    class BalancedCrossEntropyLoss(Loss):

        def __init__(self, ignore):
            123

        def get_loss(self):
            return nn.BCELoss()

    class FocalLoss:
        def __init__(self, weight, gamma):
            self.weight = weight
            self.gamma = gamma

        def get_loss(self):
            return FocalLoss(self.weight, self.gamma)


# class FocalLoss(nn.modules.loss._WeightedLoss):
#     def __init__(self, weight, gamma, device, reduction='mean'):
#         super(FocalLoss, self).__init__(weight, reduction=reduction)
#         self.gamma = gamma
#         self.weight = weight
#         self.device = device
#
#     def forward(self, pred, label):
#         """
#         preds:预测值
#         labels:真实值
#         """
#         self.weight.to(self.device)
#         pred.to(self.device)
#         label.to(self.device)
#         # pred = nn.Softmax()(pred)
#         # label = F.one_hot(np.squeeze(label.long()))[:, :, :, 0:pred.size(1)].permute(0, 3, 1, 2)
#         eps = 1e-7
#         y_pred = pred.view((pred.size()[0], pred.size()[1], -1))  # B*C*H*W->B*C*(H*W)
#         # target = label.view(y_pred.size())  # B*C*H*W->B*C*(H*W)
#         # ce = -1 * torch.log(y_pred + eps) * target
#         ce = nn.CrossEntropyLoss(ignore_index=255)(pred, label.long())
#         floss = torch.pow((1 - y_pred), self.gamma) * ce
#         floss = torch.mul(floss, self.weight)
#         floss = torch.sum(floss, dim=1)
#         return torch.mean(floss)

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, output, target):
        # 对预测值进行softmax
        output = F.softmax(output, dim=1)

        # 去掉忽略的值
        target[target == 255] = 0

        # 对标签进行one-hot编码
        target = torch.nn.functional.one_hot(target, num_classes=output.shape[1]).float().permute(0, 3, 1, 2)

        # 计算交叉熵损失
        ce_loss = F.binary_cross_entropy(output, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # 计算加权损失
        if self.alpha is not None:
            alpha = self.alpha.to(output.device)
            alpha = alpha.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(3,-1,128,128)
            focal_loss = alpha * focal_loss

        # 计算最终损失
        if self.reduction == 'mean':
            loss = focal_loss.mean()
        elif self.reduction == 'sum':
            loss = focal_loss.sum()
        else:
            loss = focal_loss

        return loss


if __name__ == "__main__":
    print(LossUtil.FocalLoss(torch.tensor([1, 1, 1]), 2).get_loss())
    # m = nn.LogSoftmax(dim=1)
    # loss = nn.NLLLoss()
    # a = torch.Tensor([0.5, 0.7, 0.1])
    # b = torch.Tensor([1])
    # print(loss(m(a), b))

    m = nn.LogSoftmax(dim=1)
    loss = nn.NLLLoss()
    # input is of size N x C = 3 x 5
    input = torch.randn(3, 5, requires_grad=True)
    imput = torch.tensor([0.2])
    # each element in target has to have 0 <= value < C
    target = torch.tensor([1, 0, 4])
    output = loss(m(input), target)
    output.backward()
