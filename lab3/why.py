from torchvision import models
import torch
import torch.onnx
import onnxruntime

mobilenet_v2 = models.mobilenet_v2(pretrained=True)

mobilenet_v2.eval()
mobilenet_v2.cuda()

###########
ONNX_FILE_PATH = 'mobilenet_v2.onnx'
input_batch = 1

torch.onnx.export(mobilenet_v2, input_batch, ONNX_FILE_PATH, export_params = True, opset_version = 10, do_constant_folding = True, input_names = ['input'], output_names = ['output'])

###########
import torchvision.transforms as transforms

ort_session = onnxruntime.InferenceSession("mobilenet_v2.onnx")
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

###############ONNX###################


