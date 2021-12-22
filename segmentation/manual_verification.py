import logging
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader
from segmentation.common import *
from segmentation.client_segmentation import IMAGE_SIZE
from segmentation.data_loader import LungSegDataset
from segmentation.models.unet import UNet
import numpy as np


def trim_ranges(l, r, bound):
    if l < 0:
        r += abs(l)
        l = 0
    if r >= bound:
        l -= (bound - r)
        r = bound - 1
    return l, r


def generate_patches(masked_image, K=10, patch_size=128):
    w, h = masked_image.shape
    shift = patch_size // 2
    patches = []
    fig = plt.figure(figsize=(10, 10))
    cols = K // 2
    rows = K // cols
    for j in range(K):
        x, y = np.where(masked_image > 0.5)
        i = np.random.randint(len(x))
        l_x, r_x = trim_ranges(x[i] - shift, x[i] + shift, w)
        l_y, r_y = trim_ranges(y[i] - shift, y[i] + shift, h)
        patches.append(masked_image[l_x:r_x, l_y:r_y])
        fig.add_subplot(rows, cols, j + 1)
        plt.imshow(patches[-1])
        plt.title(str(j))
    plt.show()
    return list(map(lambda item: torch.from_numpy(item), patches))


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)
BATCH_SIZE = 1

images_path = '../cov_dataset/images'
masks_path = '../cov_dataset/masks'
labels = "../cov_dataset/labels.csv"
model_path = '../unet_5'

net = UNet(input_channels=1,
           output_channels=64,
           n_classes=1).to(DEVICE)
net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

test_dataset = LungSegDataset(path_to_images=images_path,
                              path_to_masks=masks_path,
                              image_size=IMAGE_SIZE,
                              mode="test", labels=labels)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

criterion = DiceLoss()
net.eval()
val_running_loss = 0.0
val_running_jac = 0.0

for batch_idx, (images, masks) in enumerate(test_loader):
    images = images.to(DEVICE)
    masks = masks.to(DEVICE)
    outputs_masks = net(images)
    # loss_seg = criterion(outputs_masks, masks)
    # loss = loss_seg

    # val_running_loss += loss.item()
    # jac = jaccard(outputs_masks.round(), masks)
    # val_running_jac += jac.item()
    img = images[0, 0, :]
    mask = masks[0, 0, :]
    img_np = img.numpy()
    out = outputs_masks[0, 0, :]
    out_np = out.detach().numpy()
    # superposed = cv2.bitwise_and(img.numpy(), out_np)
    superposed = np.copy(img_np)
    superposed[out_np < 0.05] = 0
    res = torch.cat((img, mask, out, torch.from_numpy(superposed)), 1).cpu().detach()
    plt.imshow(res)
    plt.show()
    generate_patches(superposed)

val_loss = val_running_loss / len(test_loader)
val_jac = val_running_jac / len(test_loader)
