import os

import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from segmentation.client_segmentation import IMAGE_SIZE
from segmentation.common import *
from segmentation.models.unet import UNet
import cv2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)
BATCH_SIZE = 1

images_path = './to_cut/'
model_path = '../unet_5'

net = UNet(input_channels=1,
           output_channels=64,
           n_classes=1).to(DEVICE)
net.load_state_dict(torch.load(model_path, map_location=DEVICE))

for img in os.listdir(images_path):
    img_path = os.path.join(images_path, img)
    print(os.getcwd())
    print(img_path)
    image = Image.open(img_path).convert("L")
    resize_transform = transforms.Resize(size=(IMAGE_SIZE, IMAGE_SIZE))
    image = resize_transform(image)
    image = F.to_tensor(image)
    input_batch = image.unsqueeze(0)
    out = net(input_batch)
    out = out[0, 0, :]
    input = input_batch[0, 0, :]
    res = torch.cat((input, out), 1).cpu().detach()
    plt.imshow(res)
    plt.show()
