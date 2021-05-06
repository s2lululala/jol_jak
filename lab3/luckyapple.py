# 필요한 import문
import io
import numpy as np

from torch import nn
#import torch.utils.model_zoo as model_zoo
import torch.onnx

import sys


print(sys.getrecursionlimit())

sys.setrecursionlimit(10000)

######################################################################

import torch.nn as nn
import torch.nn.init as init
from torchvision import models

ONNX_FILE_PATH = "mobilenet_v2.onnx"

mobilenet_v2  = models.mobilenet_v2(pretrained = True)

#model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
batch_size = 1
map_location = lambda storage, loc: storage

if torch.cuda.is_available():
    map_location = None

#mobilenet_v2.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))

mobilenet_v2.eval()


x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
torch_out = mobilenet_v2(x)
#mobilenet_v2.cuda()

torch.onnx.export(mobilenet_v2, x, ONNX_FILE_PATH, export_params = True, opset_version = 10, do_constant_folding = True, input_names = ['input'], output_names = ['output'])

import onnx

onnx_model = onnx.load(ONNX_FILE_PATH)
onnx.checker.check_model(onnx_model)

import tensorrt as trt
TRT_LOGGER = trt.Logger()
trt_file_name = "mobilenet_v2.plan"
def build_endgine(onnx_file_path):
    bulder = trt.Builder(TRT_LOGGER)
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = bulder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file_path, 'rb') as model:
        print("Beginning ONNX file parsing")
        parser.parse(model.read())

    last_layer = network.get_layer(network.num_layers - 1)
    network.mark_output(last_layer.get_output(0))
    print("Completed parsing of ONNX file")

import onnxruntime

ort_session = onnxruntime.InferenceSession(ONNX_FILE_PATH)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()    
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")

from PIL import Image
import torchvision.transforms as transforms

img = Image.open("./images/fruit_1.jpg")

resize = transforms.Resize([224, 224])
img = resize(img)

img_ycbcr = img.convert('YCbCr')
img_y, img_cb, img_cr = img_ycbcr.split()

to_tensor = transforms.ToTensor()
img_y = to_tensor(img_y)
img_y.unsqueeze_(0)

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
ort_outs = ort_session.run(None, ort_inputs)
img_out_y = ort_outs[0]

img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')

final_img = Image.merge(
    "YCbCr", [
        img_out_y,
        img_cb.resize(img_out_y.size, Image.BICUBIC),
        img_cr.resize(img_out_y.size, Image.BICUBIC),
    ]).convert("RGB")

final_img.save("./result/out_fruit.jpg")
