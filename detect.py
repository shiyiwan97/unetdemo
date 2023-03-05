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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        unet = net.UNet().to(device)
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

        for i, (image, segment_image) in enumerate(tqdm.tqdm(test_data)):
            image, segment_image = image.to(device), segment_image.to(device)
            a1 = image[0].cpu()
            a2 = np.array(a1)
            a3 = T.ToPILImage()(a1)
            # a3.show()
            out_image = unet(image)
            # softmaxResult = F.softmax(out_image, dim=1).type(torch.FloatTensor)[0].permute(1, 2, 0)
            softmaxResult = out_image[0].permute(1, 2, 0)
            result = self.convertToL(self, softmaxResult, 128, 128)

            image = T.ToPILImage()(result)
            image.show()

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

    method = 1

    if (method == 1):
        Detect.detect(r'C:\\Users\\shiyiwan\\Desktop\\1', r'C:\\Users\\shiyiwan\\Desktop\\1\\detect',
                      'weight/latest/weight_latest.pth')

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
