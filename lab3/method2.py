import torch
import tensorrt as trt
from torch2trt import torch2trt ###
from torchvision import models

model = models.mobilenet_v2(pretrained=True).eval()

filename = ("lemon.jpg") # plz change the img file name!

# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')
    model_trt = torch2trt(model, [input_batch], log_level=trt.Logger.Severity.VERBOSE) ###

with torch.no_grad():
    output = model(input_batch)
    output_trt = model_trt(input_batch) ###

# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes

# The output has unnormalized scores. To get probabilities, you can run a softmax on it.

# --- pytorch --- #
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())

# --- tensortrt --- #
probabilities_trt = torch.nn.functional.softmax(output_trt[0], dim=0)

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories_trt = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob_trt, top5_catid_trt = torch.topk(probabilities_trt, 5)
for i in range(top5_prob_trt.size(0)):
    print(categories_trt[top5_catid_trt[i]], top5_prob_trt[i].item())

print(torch.max(torch.abs(output - output_trt))) ### almost 0

# save
torch.save(model.state_dict(), 'mobilenetv2.pth')
torch.save(model_trt.state_dict(), 'mobilenetv2_trt.pth')

# load
'''
from torch2trt import TRTModule

model_trt = TRTModule()

model_trt.load_state_dict(torch.load('mobilenetv2_trt.pth'))
'''
