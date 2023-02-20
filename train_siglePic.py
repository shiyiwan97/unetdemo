import os.path

from torch.utils.data import DataLoader
from data import *
import tqdm
from net.net import *
from torch import nn, optim
import torch.nn.functional as F


def TrainSinglePic():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data_path = 'F:\machineLearning\dataset\dataset_1000'
    test_data_path = ''
    epoch_num = 200

    train_data = DataLoader(MyDataset(train_data_path))
    # test_data = DataLoader(MyDataset(test_data_path))

    weight_path = 'weight/weight.pth'

    unet = UNet().to(device)
    if os.path.exists((weight_path)):
        unet.load_state_dict(torch.load(weight_path))

    loss = nn.CrossEntropyLoss()
    opt = optim.SGD(unet.parameters(), lr=0.01)
    # todo data应该是返回一组数据，train+test

    if os.path.exists(weight_path):
        unet.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')

    epoch = 1
    while epoch < 200:
        for i, (image, segment_image) in enumerate(tqdm.tqdm(train_data)):
            image, segment_image = image.to(device), segment_image.to(device)
            out_image = unet(image)

            train_loss = loss(
                F.softmax(out_image, dim=1).type(torch.FloatTensor),
                F.one_hot(np.squeeze(torch.where(segment_image == 255, 0, segment_image), axis=1).long(), 19)
                .permute(0, 3, 2, 1).float()
            )
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            if i % 1 == 0:
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')
            if i % 20 == 0:
                torch.save(unet.state_dict(), weight_path)
                # 保存权重
                print('save successfully!')

        epoch += 1


if __name__ == '__main__':
    TrainSinglePic()
