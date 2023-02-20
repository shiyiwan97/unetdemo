from net import net
import torch
import os
from utils.commonUtil import *
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
            unet.load_state_dict(torch.load(weightPath))
        else:
            raise Exception('读取权重数据失败！')

        picPathList = CommonUtil.getUriList(picPath, ".png")
        test_data = DataLoader(MyDataset(picPath))

        for i, (image, segment_image) in enumerate(tqdm.tqdm(test_data)):
            image, segment_image = image.to(device), segment_image.to(device)
            out_image = unet(image)
            softmaxResult = F.softmax(out_image, dim=1).type(torch.FloatTensor)[0].permute(1, 2, 0)
            result = self.convertToL(self, softmaxResult, 512, 512)

            image = T.ToPILImage()(result)
            image.show()

            # todo 多分类问题，softmax到灰度图
            print(123)

    # 多分类结果转换为灰度图
    # [512,512,19] -> [1,512,512]
    # todo 宽高自己算
    def convertToL(self, tensor, width, height):

        result = torch.empty(1, width, height)

        for i in range(width):
            for j in range(height):
                result[0, i, j] = torch.max(tensor[i, j], dim=0)[1] + 1

        return result


if __name__ == '__main__':

    method = 1

    if (method == 1):
        Detect.detect('F:\machineLearning\dataset\\test', 'F:\machineLearning\dataset\\test\\result',
                      'weight/weight.pth')

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
