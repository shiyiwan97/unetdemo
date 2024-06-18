import numpy as np
import torch
from PIL import Image
import cv2

"""
1. ndarray先变换再用torch_from_numpy与先用torch.tensor再变换得到的结果是一样的
"""
def test1():

    image = cv2.imread(r'D:\Dataset\dataset_cpu_1\000000.png', cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    image_nd2t = torch.from_numpy(image)

    image = cv2.imread(r'D:\Dataset\dataset_cpu_1\000000.png', cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.tensor((image / 255.0), dtype=torch.float)
    image = image.permute(2, 0, 1)
    image_t = image.unsqueeze(0)

    np.testing.assert_allclose(image_nd2t,image_t,0,0)
    print("pass")

def main():
    test1()

if __name__ == '__main__':
    main()