import torch
import torch.nn as nn

# 定义一个简单的神经网络模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 5)  # 一个简单的全连接层，输入特征为10，输出特征为5

    def forward(self, x):
        return self.fc(x)

# 创建模型的实例
model = SimpleNet()

# 创建一个模拟输入
example_input = torch.rand(1, 10)  # batch size = 1, 输入特征为10

# 追踪模型来产生Torch Script
traced_model = torch.jit.trace(model, example_input)

# 保存追踪的模型
traced_model.save("simple_model.pt")
    