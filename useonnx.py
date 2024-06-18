import cv2
import onnxruntime as ort
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from net import net

def main():
    # 加载模型
    session = ort.InferenceSession("unet.onnx")


    # 输入图像
    image = cv2.imread(r'D:\Dataset\dataset_cpu_1\000000.png', cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    # 调整图像通道顺序为RGB到CxHxW
    image = np.transpose(image, (2, 0, 1))
    # 添加批次维度
    image = np.expand_dims(image, axis=0)


    #先转tensor再


    # 获取模型输入输出
    input_name = session.get_inputs()[0].name
    # [expression for item in iterable]，语法糖，把遍历后的元素执行计算表达式expression后组成新的列表
    output_names = [output.name for output in session.get_outputs()]
    # 进行推理
    out_numpy = session.run(output_names, {input_name: image})
    print("onnx输入：" + str(torch.from_numpy(image).sum()))
    out_image = torch.from_numpy(out_numpy[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = net.UNet().to(device)
    unet.load_state_dict(torch.load('weight\\latest\\weight_latest.pth'))
    # unet.eval()
    input_tensor = torch.from_numpy(image)

    # 之前的inputtensor
    image = cv2.imread(r'D:\Dataset\dataset_cpu_1\000000.png', cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将灰度图像转换为彩色图像
    image = torch.tensor((image / 255.0), dtype=torch.float)
    image = image.permute(2, 0, 1)
    image = image.unsqueeze(0)

    equal = torch.equal(image, input_tensor)
    print(equal)

    print("unet输入：" + str(input_tensor.sum()))
    output_unet = unet(input_tensor)
    out_image = output_unet
######################################

    # 在第二个维度进行softmax，得到的是一个[batchsize,19,128,128]的tensor，其中[batchsize,:,0,0].sum()=1
    softmax = F.softmax(out_image, dim=1)
    # 转化为32位浮点数张量，并且取出第一个维度，就是batchsize的第一张，此时为[19,128,128]
    softmaxResult = softmax.type(torch.FloatTensor)[0]
    print("softmaxresult:")
    print(softmaxResult.size())
    print("-------------------")
    print(softmaxResult[0][0])
    # 进行argmax，得到的是一个[128,128]的张量，值为每个像素点预测的类别
    pred = torch.argmax(softmaxResult, dim=0)
    # 为了让显示更加明显，放大多倍
    resizePred = pred.float() / 18 * 255
    # 转化为uint8，能更好的兼容图像显示
    unit8Pic = resizePred.type(torch.uint8)
    # 显示图片
    result = T.ToPILImage()(unit8Pic)
    result.show()

if __name__ == '__main__':
    main()