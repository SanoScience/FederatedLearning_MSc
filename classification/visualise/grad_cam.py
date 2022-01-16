from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import torch
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

import PIL
import time

st = time.time()

model = resnet50(pretrained=True)
model.fc = torch.nn.Linear(in_features=2048, out_features=3)
model.load_state_dict(torch.load('./ResNet50_9_acc_0.747_loss_0.61', map_location=torch.device('cpu')))
# model.load_state_dict(torch.load('./ResNet50_5_acc_0.734_loss_0.607', map_location=torch.device('cpu')))

target_layer = model.layer4[-1]
print(target_layer)
cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=False)

# image_rgb = PIL.Image.open('./Lung_Opacity-1.png').convert('RGB')
# image_rgb = PIL.Image.open('./Lung_Opacity-210.png').convert('RGB')
# image_rgb = PIL.Image.open('./Normal-1.png').convert('RGB')
image_rgb = PIL.Image.open('./original-ffc01e64-ba14-4620-8016-235fc1609767.png').convert('RGB')
# image_rgb = PIL.Image.open('./ffc01e64-ba14-4620-8016-235fc1609767.png').convert('RGB')
# image_rgb = PIL.Image.open('./002cb550-2e31-42f1-a29d-fbc279977e71.png').convert('RGB')
# image_rgb = PIL.Image.open('./masked.png').convert('RGB')

convert = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.ToTensor(),
    # Normalize to ImageNet
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])

input_tensor = convert(image_rgb).float()

v = Variable(input_tensor, requires_grad=True)
v = v.unsqueeze(0)

grayscale_cam = cam(input_tensor=v)

res_conv = torchvision.transforms.Resize(size=(224, 224))

image_rgb = res_conv(image_rgb)

img_np = np.array(image_rgb) / 255

visualization = show_cam_on_image(img_np, grayscale_cam[0], use_rgb=True)
plt.imshow(visualization)
plt.show()

et = time.time()

print(et-st)