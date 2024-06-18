import onnxruntime as ort
import numpy as np
from collections import OrderedDict
from config import *
from PIL import Image
from numpy import mean
import time
import datetime

def composed_transforms(image):
    mean = np.array([0.485, 0.456, 0.406])  # 均值
    std = np.array([0.229, 0.224, 0.225])  # 标准差
    # transforms.Resize是双线性插值
    resized_image = image.resize((args['scale'], args['scale']), resample=Image.BILINEAR)
    # onnx模型的输入必须是np,并且数据类型与onnx模型要求的数据类型保持一致
    resized_image = np.array(resized_image)
    normalized_image = (resized_image/255.0 - mean) / std
    return np.round(normalized_image.astype(np.float32), 4)

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

to_test = OrderedDict([
                       # ('CHAMELEON', chameleon_path),
                       # ('CAMO', camo_path),
                       ('COD10K', cod10k_path),
                       ])
args = {
    'scale': 416,
    'save_results': True
}

def main():
    # 保存检测结果的地址
    results_path = './results2'
    exp_name = 'PFNet'
    providers = ["CPUxecutionProvider"]
    ort_session = ort.InferenceSession("PFNet.onnx", providers=providers)  # 创建一个推理session
    input_name = ort_session.get_inputs()[0].name
    # 输出有四个
    output_names = [output.name for output in ort_session.get_outputs()]
    start = time.time()
    for name, root in to_test.items():
        time_list = []
        image_path = os.path.join(root, 'image')
        if args['save_results']:
            check_mkdir(os.path.join(results_path, exp_name, name))
        img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('jpg')]
        for idx, img_name in enumerate(img_list):
            img = Image.open(os.path.join(image_path, img_name + '.jpg')).convert('RGB')
            w, h = img.size
            #  对原始图像resize和归一化
            img_var = composed_transforms(img)
            # np的shape从[w,h,c]=>[c,w,h]
            img_var = np.transpose(img_var, (2, 0, 1))
            # 增加数据的维度[c,w,h]=>[bathsize,c,w,h]
            img_var = np.expand_dims(img_var, axis=0)
            start_each = time.time()
            prediction = ort_session.run(output_names, {input_name: img_var})
            time_each = time.time() - start_each
            time_list.append(time_each)
            # 除去多余的bathsize维度,NumPy变会PIL同样需要变换数据类型
            # *255替换pytorch的to_pil
            prediction = (np.squeeze(prediction[3])*255).astype(np.uint8)
            if args['save_results']:
               (Image.fromarray(prediction).resize((w, h)).convert('L').save(os.path.join(results_path, exp_name, name, img_name + '.png')))
        print(('{}'.format(exp_name)))
        print("{}'s average Time Is : {:.3f} s".format(name, mean(time_list)))
        print("{}'s average Time Is : {:.1f} fps".format(name, 1 / mean(time_list)))
    end = time.time()
    print("Total Testing Time: {}".format(str(datetime.timedelta(seconds=int(end - start)))))
if __name__ == '__main__':
    main()
