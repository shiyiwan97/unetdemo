import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary


class ConvBNReLUTwice(nn.Module):
    """(conv => BN => ReLU)*2"""
    kernel_size = 3
    stride = 1
    padding = 1
    padding_mode = 'reflect'
    bias = False

    def __init__(self, in_channels, out_channels):
        # super().__init__()类似于java中的super.init，调用父类中的init方法
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, self.kernel_size, self.stride, self.padding,
                      padding_mode=self.padding_mode, bias=self.bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, self.kernel_size, self.stride, self.padding,
                      padding_mode=self.padding_mode, bias=self.bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


class Pool(nn.Module):
    """maxPool"""
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 0

    def __init__(self, channel):
        super(Pool, self).__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(self.kernel_size, self.stride)
        )

    def forward(self, x):
        return self.layer(x)


class Up(nn.Module):
    def __init__(self, channel):
        super(Up, self).__init__()
        self.conv = nn.Conv2d(channel, channel // 2, 1, 1)

    def forward(self, x1, x2):
        up = F.interpolate(x1, scale_factor=2, mode='nearest')
        conv = self.conv(up)
        return torch.cat((conv, x2), dim=1)


class UpAndConcat(nn.Module):
    def __init__(self, channel):
        super(UpAndConcat, self).__init__()
        self.conv = nn.Conv2d(channel, channel // 2, 1, 1)

    def forward(self, x1, x2):
        up = F.interpolate(x1, scale_factor=2, mode='nearest')
        conv = self.conv(up)
        return torch.cat((conv, x2), dim=1)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.c1 = ConvBNReLUTwice(3, 64)
        self.p1 = Pool(64)
        self.c2 = ConvBNReLUTwice(64, 128)
        self.p2 = Pool(128)
        self.c3 = ConvBNReLUTwice(128, 256)
        # self.u1 = Up(256)
        self.u1 = UpAndConcat(256)
        self.c4 = ConvBNReLUTwice(256, 128)
        # self.u2 = Up(128)
        self.u2 = UpAndConcat(128)
        self.c5 = ConvBNReLUTwice(128, 64)
        self.out = nn.Conv2d(64, 19, 3, 1, 1)

    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.p1(R1))
        R3 = self.c3(self.p2(R2))
        R4 = self.c4(self.u1(R3, R2))
        R5 = self.c5(self.u2(R4, R1))

        return self.out(R5)


if __name__ == '__main__':
    # x = torch.randn(1, 3, 128, 128)
    unet = UNet()
    # print(net(x).shape)
    print(summary(unet.cuda(),(3,128,128)))
