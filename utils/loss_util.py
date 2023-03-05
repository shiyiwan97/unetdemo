from abc import abstractmethod

import torch
from torch import nn


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
            return FocalLoss(self.weight,self.gamma)

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight

    def forward(self, preds, labels):
        """
        preds:softmax输出结果
        labels:真实值
        """
        eps = 1e-7
        y_pred = preds.view((preds.size()[0], preds.size()[1], -1))  # B*C*H*W->B*C*(H*W)
        target = labels.view(y_pred.size())  # B*C*H*W->B*C*(H*W)
        ce = -1 * torch.log(y_pred + eps) * target
        floss = torch.pow((1 - y_pred), self.gamma) * ce
        floss = torch.mul(floss, self.weight)
        floss = torch.sum(floss, dim=1)
        return torch.mean(floss)


if __name__ == "__main__":

    print(LossUtil.FocalLoss(torch.tensor([1,1,1]),2).get_loss())
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