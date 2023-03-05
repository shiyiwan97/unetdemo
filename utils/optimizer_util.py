from abc import abstractmethod

from torch import optim


class OptimizerUtil:
    """
    优化器工具
    """

    class Optimizer:
        """
        优化器抽象父类
        """

        @abstractmethod
        def get_optimizer(self):
            """
            获取优化器
            """
            pass

    class SGD(Optimizer):
        """
        SGD优化器
        """

        def __init__(self, lr, momentum, weight_decay):
            self.lr = lr
            self.momentum = momentum
            self.weight_decay = weight_decay

        def get_optimizer(self, parameters):
            return optim.SGD(parameters, self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
