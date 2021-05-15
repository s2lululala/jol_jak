import torch
import onnx
from torchvision import models

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()

model = models.mobilenet_v2(pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
model.eval()

filename = ("fruit_0.jpg") # plz change the img file name!

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

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes

# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)

end.record()
torch.cuda.synchronize()
print("model forward time : " + str(start.elapsed_time(end) / 1000))

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())

ONNX_FILE_PATH = "mobilenet_v2.onnx"
torch.onnx.export(model, input_batch, ONNX_FILE_PATH, input_names = ["input"], output_names = ["output"], export_params = True)
onnx_model = onnx.load(ONNX_FILE_PATH)
onnx.checker.check_model(onnx_model)

print("It was saved to", ONNX_FILE_PATH)
