from net import net
import torch
import os
from utils.common_util import *
from data import *
from torch.utils.data import DataLoader
import tqdm
import torch.nn.functional as F
import torchvision.transforms as T


class Detect:

    @classmethod
    def detect(self, picPath, resultPath, weightPath):
        # with torch.no_grad():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            unet = net.UNet().to(device)

            use_hook = True

            if use_hook:
                def hook_fn(module, input, output):
                    print(module)
                    print('hook input[0][0][0]:', input[0][0][0])
                    print('hook output[0][0][0]:', output[0][9][256])

                unet.c1.register_forward_hook(hook_fn)


            if os.path.exists((weightPath)):
                # unet = torch.load(weightPath, map_location=torch.device('cpu'))
                if (device == torch.device('cpu')):
                    unet.load_state_dict(torch.load(weightPath, map_location=torch.device('cpu')))
                else:
                    unet.load_state_dict(torch.load(weightPath))
            else:
                raise Exception('读取权重数据失败！')

            picPathList = CommonUtil.getUriList(picPath, ".png")
            test_data = DataLoader(MyDataset(picPath))
            # unet.eval()
            for i, (image, segment_image) in enumerate(tqdm.tqdm(test_data)):
                image, segment_image = image.to(device), segment_image.to(device)
                a1 = image[0].cpu()
                a2 = np.array(a1)
                a3 = T.ToPILImage()(a1)
                # a3.show()
                # 输出的是一个[batchsize,19,128,128]的tensor

                image = cv2.imread(r'D:\Dataset\dataset_cpu_1\000000.png', cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将灰度图像转换为彩色图像
                image = torch.tensor((image / 255.0), dtype=torch.float)
                image = image.permute(2, 0, 1)
                image = image.unsqueeze(0)


                print("输入模型的张量image:")
                print(image.size())
                print(image)
                out_image = unet(image)
                print("输出模型的张量out_image")
                print(out_image.size())
                print(out_image[0][0][0])
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
                # softmaxResult = out_image[0].permute(1, 2, 0)
                # result = self.convertToL(self, softmaxResult, 128, 128)
                #
                # image = T.ToPILImage()(result)
                # image.show()

                # todo 多分类问题，softmax到灰度图
                print(123)

    # 多分类结果转换为灰度图
    # [128,128,19] -> [1,128,128]
    # todo 宽高自己算
    def convertToL(self, tensor, width, height):

        result = torch.empty(1, width, height)

        for i in range(width):
            for j in range(height):
                result[0, i, j] = torch.max(tensor[i, j], dim=0)[1]

        nparray = np.array(result.permute(1, 2, 0))
        return nparray


if __name__ == '__main__':

    # for i in range(2, 4):
    #     print(i)

    # method = 3
    method = 3

    if (method == 1):
        Detect.detect(r'D:\Dataset\dataset_1000\test1', r'D:\Dataset\dataset_1000\test1\rs',
                      'weight\\baseline\\weight_2.pth')

    if (method == 3):
        Detect.detect(r'D:\Dataset\dataset_cpu_1', r'D:\Dataset\dataset_cpu_1\rs',
                      'weight\\latest\\weight_latest.pth')

    if (method == 5):
        Detect.detect(r'D:\Dataset\dataset_cpu_1', r'D:\Dataset\dataset_cpu_1\rs',
                      'weight\\latest\\weight_latest.pth')

    if (method == 2):
        lImg = Image.open("F:\machineLearning\dataset\\test\\000000_seg.png")

        tensor = T.ToTensor()(lImg)
        tensor = torch.from_numpy(np.array(lImg))
        # t1 = F.one_hot(np.squeeze(torch.where(tensor == 255, 0, tensor), axis=1).long(), 19).permute(0, 2, 1, 3).type(
        #     torch.FloatTensor)
        # t2 = Detect().convertToL(t1[0], 512, 512)
        # t3 = T.ToPILImage()(t2)
        # t3.show()

        a = F.one_hot(np.squeeze(torch.where(tensor == 255, 0, tensor), axis=1).long(), 19)
        t2 = Detect().convertToL(a, 512, 512)
        t3 = T.ToPILImage()(t2)
        t3.show()

    if (method == 4):
        model = torch.load("model.pt")
        print(model)
        print("-----------------------")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(net.UNet().to(device))
