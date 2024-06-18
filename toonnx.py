import onnx
from onnx import numpy_helper
import torch
from net import net
import numpy as np
import onnxruntime as ort

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet = net.UNet().to(device)

unet.load_state_dict(torch.load('weight\\latest\\weight_latest.pth'))
unet.eval()

np_example = np.random.rand(1,3,512,512).astype(np.float32)
tensor_example = torch.from_numpy(np_example)


torch.onnx.export(unet, tensor_example, r"unet.onnx")
model_onnx = onnx.load(r"unet.onnx")
onnx.checker.check_model(model_onnx)
print(onnx.helper.printable_graph(model_onnx.graph))


# pytorch输出
output_pytorch = unet(tensor_example)
# onnx输出
session = ort.InferenceSession(r"unet.onnx")
input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]
output_onnx = session.run(None,{input_name:np_example})[0]
from_numpy = torch.from_numpy(output_onnx)
equal = torch.equal(output_pytorch, from_numpy)
print(equal)

output_pytorch_np = output_pytorch.detach().numpy()
np.testing.assert_allclose(output_pytorch_np,output_onnx,rtol=1e-3,atol=1e-5)
print("pass")


