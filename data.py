import os

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor()
    # torch.from_numpy()
])

class MyDataset(Dataset):

    def __init__(self,path):
        # self.data = os.listdir(path)
        self.path = path
        fileList = os.listdir(path)
        self.fileNameList = []
        for i in range(len(fileList)):
            if (fileList[i].__contains__("_seg.png")):
                self.fileNameList.append(fileList[i][:-8])

    def __len__(self):
        return len(self.fileNameList)

    def __getitem__(self, index):
        rawPic = Image.open(os.path.join(self.path,self.fileNameList[index] + '.png'))
        segPic = Image.open(os.path.join(self.path,self.fileNameList[index] + '_seg.png'))
        # return torch.from_numpy(np.array(rawPic)),torch.from_numpy(np.array(segPic))
        return transform(rawPic),torch.from_numpy(np.array(segPic))

if __name__ == '__main__':
    # np.set_printoptions(threshold=np.inf)

    img1 = Image.open('F:\machineLearning\dataset\dataset_1000\\000000_seg.png')
    img2 = Image.open('F:\machineLearning\dataset\dataset_1000\\000000_seg.png')
    np.array(img2)

    a = MyDataset('F:\machineLearning\dataset\dataset_1000').__getitem__(0)[1]

    print(a)
    print(torch.from_numpy(np.array(img2)))


