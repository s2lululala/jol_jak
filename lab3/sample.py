'''
https://github.com/NVIDIA-AI-IOT/torch2trt
'''

# ---------- 1. convert ---------- #
import torch
from torch2trt import torch2trt
from torchvision.models.alexnet import alexnet

# create some regular pytorch model...
model = alexnet(pretrained=True).eval().cuda()

# create example data
x = torch.ones((1, 3, 224, 224)).cuda()

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x])
print('=======1======')
# ---------- 2. execute ---------- #
y = model(x)
y_trt = model_trt(x)

# check the output against PyTorch
print(torch.max(torch.abs(y - y_trt)))
print('=======2======')

# ---------- 3. save ---------- #
torch.save(model_trt.state_dict(), 'alexnet_trt.pth')

# ---------- 4. load ---------- #

from torch2trt import TRTModule

model_trt = TRTModule()

model_trt.load_state_dict(torch.load('alexnet_trt.pth'))

# ---------- 5. ---------- #
# shell, ./test.sh sample.py
