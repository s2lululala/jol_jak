from torchvision import models
import torch
import torch.onnx

resnet50 = models.resnet50(pretrained=True)
resnet50.eval()

ONNX_FILE_PATH = 'resnet50.onnx'

input_batch = 1

torch.onnx.export(resnet50, input_batch, ONNX_FILE_PATH, export_params = True, opset_version = 10, do_constant_folding = True, input_names = ['input'], output_names = ['output'])
