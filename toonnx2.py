import cv2
import onnx
from onnx import numpy_helper
import torch

from net import net
import numpy as np
import onnxruntime as ort
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as T

session = None
input_name = None
init = False

def main():

    unet = getEvalUnet()
    npArray = getPicNp()
    tensor_example = torch.from_numpy(npArray)
    torch.onnx.export(unet, tensor_example, r"unet2.onnx")

    # model_onnx = onnx.load(r"unet.onnx")
    # onnx.checker.check_model(model_onnx)
    # print(onnx.helper.printable_graph(model_onnx.graph))


    # pytorch eval输出
    unet.eval()
    output_pytorch = unet(tensor_example)
    # pytorch train输出
    unet.train()
    output_pytorch_train = unet(tensor_example)
    # onnx输出
    output_onnx = detectOnnx(npArray)
    from_numpy = torch.from_numpy(output_onnx)

    # postProcess(True,output_onnx)
    postProcess(False,output_pytorch)
    postProcess(False,output_pytorch_train)


    # equal = torch.equal(output_pytorch, from_numpy)
    # print(equal)
    #
    # output_pytorch_np = output_pytorch.detach().numpy()
    # print(str(output_pytorch.sum()) + "---" + str(from_numpy.sum()))
    # np.testing.assert_allclose(output_pytorch_np,output_onnx,rtol=1e-3,atol=1e-5)
    # print("pass")


def getPicNp():
    # 使用Pillow库读取图像
    img = Image.open(r'D:\Dataset\dataset_cpu_1\000000.png')
    # 将PIL图像转换成RGB格式（如果图像不是RGB格式）
    img = img.convert('RGB')
    # 将PIL图像转换为numpy数组，并且将数据类型转换为float32
    image = np.array(img, dtype=np.float32) / 255.0
    # 转置数组的维度，使其从HWC变为CHW格式
    image = np.transpose(image, (2, 0, 1))
    # 在数组前增加一个新的维度，用于批处理
    image = np.expand_dims(image, axis=0)
    return image

#后处理
def postProcess(isNumpy,output):
    if isNumpy:
        out_image = torch.from_numpy(output)
    else:
        out_image = output

    # 在第二个维度进行softmax，得到的是一个[batchsize,19,128,128]的tensor，其中[batchsize,:,0,0].sum()=1
    softmax = F.softmax(out_image, dim=1)
    # 转化为32位浮点数张量，并且取出第一个维度，就是batchsize的第一张，此时为[19,128,128]
    softmaxResult = softmax.type(torch.FloatTensor)[0]
    # 进行argmax，得到的是一个[128,128]的张量，值为每个像素点预测的类别
    pred = torch.argmax(softmaxResult, dim=0)
    # 为了让显示更加明显，放大多倍
    resizePred = pred.float() / 18 * 255
    # 转化为uint8，能更好的兼容图像显示
    unit8Pic = resizePred.type(torch.uint8)
    # 显示图片
    result = T.ToPILImage()(unit8Pic)
    result.show()

def getEvalUnet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = net.UNet().to(device)
    unet.load_state_dict(torch.load('weight\\latest\\weight_latest.pth'))
    unet.eval()
    return unet

def initOnnx():
    global session
    global input_name
    global init
    session = ort.InferenceSession(r"unet2.onnx")
    input_name = session.get_inputs()[0].name
    init = True


def detectOnnx(imageArray):
    if not init:
        initOnnx()
    out_numpy = session.run(None, {input_name: imageArray})
    return out_numpy[0]

if __name__ == '__main__':
    main()
