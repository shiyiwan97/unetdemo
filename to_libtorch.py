import torch
import cv2
from net import net

if __name__ == '__main__':
    method = 1;
    weight_path = 'weight\\latest\\weight_latest.pth'
    with torch.no_grad():  # 不跟踪梯度
        if (method == 1):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print("device:" + str(device))
            unet = net.UNet().to(device)
            unet.load_state_dict(torch.load(weight_path, map_location=device))
            # 确认模型进入评估模式，不启用dropout和batch normalization等推理中不需要的训练特定行为
            # unet.eval()
            example = torch.rand(1, 3, 512, 512)
            traced_script_module = torch.jit.trace(unet, example)
            traced_script_module.save('model.pt')

            # 测试这个pt能调用不
            model = torch.jit.load('model.pt')
            model.eval()
            image = cv2.imread(r'D:\Dataset\dataset_cpu_1\000000.png', cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将灰度图像转换为彩色图像
            image = torch.tensor((image/ 255.0), dtype=torch.float)
            image = image.permute(2, 0, 1)
            image = image.unsqueeze(0)
            # print(image)# 调整通道顺序为 (3, 512, 512)
            output = model(image)
            print("----------------")
            # print(output)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            unet = net.UNet().to(device)
            weightPath = 'weight\\latest\\weight_latest.pth'
            unet.load_state_dict(torch.load(weightPath, map_location=torch.device('cpu')))
            #unet = torch.load('weight\\latest\\weight_latest.pth')
            unet.eval()
            print(unet)

            print(model)


            # print(model)
            #print(unet.items()[0])
            print("unet ..............................")
            for name, param in unet.state_dict().items():
                print(f"name={name}, size={param.size()} ,sum={torch.sum(param)}")



            print("pt ..............................")
            for name, module in model.named_modules():
                print(f"{name}:")
                # 打印模块的参数
                for param_name, param in module.named_parameters(recurse=False):
                    print(f"  {param_name}: {param.size()},sum={torch.sum(param)}")
                #for sub_name, sub_module in module.named_children():
                #    print(f"\n{name}.{sub_name}:")
                #    for sub_param_name, sub_param in sub_module.named_parameters(recurse=False):
                #        print(f"  {sub_param_name}: {param.size()},sum={torch.sum(param)}")

            output1 = unet(image)

            print(output)
            print(output1)
