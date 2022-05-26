import os

import albumentations as albu
from albumentations.pytorch import ToTensorV2
import argparse
import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torchvision import models
from torchvision import transforms
import time
import os
import cv2
import numpy as np
from sklearn.metrics import classification_report

from data_loader import NIHDataset
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from classification_covid.efnetv2 import effnetv2_m
from efficientnet_pytorch import EfficientNet


# from comet_ml import Experiment
# experiment = Experiment(api_key="U3FlRg7XdZldXwLF3TLe8hqSP", project_name="covid-cxr",
#                        workspace="twlodarczyk")

def accuracy_score(pred, actual):
    correct = 0
    act_labels = actual == 1
    same = act_labels == pred
    correct = same.sum().item()
    total = actual.shape[0] * actual.shape[1]
    return correct / total


def get_ENS_weights(num_classes, samples_per_class, beta=1):
    ens = 1.0 - np.power([beta] * num_classes, np.array(samples_per_class, dtype=np.float))
    weights = (1.0 - beta) / np.array(ens)
    weights = weights / np.sum(weights) * num_classes
    return torch.as_tensor(weights, dtype=torch.float)


parser = argparse.ArgumentParser(description="Train classifier to detect covid on CXR images.")

DATASET_PATH_BASE = os.path.expandvars("$SCRATCH/fl_msc/classification/NIH/data/")

parser.add_argument("--images",
                    type=str,
                    default=os.path.join(DATASET_PATH_BASE, "images/"),
                    help="Path to the images")
parser.add_argument("--labels",
                    type=str,
                    default=os.path.join(DATASET_PATH_BASE, "labels/nih_data_labels.csv"),
                    help="Path to the labels")
parser.add_argument("--train_subset",
                    type=str,
                    default=os.path.join(DATASET_PATH_BASE, "partitions/nih_train_val_list.txt"),
                    help="Path to the file with training dataset files list")
parser.add_argument("--test_subset",
                    type=str,
                    default=os.path.join(DATASET_PATH_BASE, "partitions/nih_test_list.txt"),
                    help="Path to the file with test/validation dataset files list")
parser.add_argument("--in_channels",
                    type=int,
                    default=3,
                    help="Number of input channels")
parser.add_argument("--epochs",
                    type=int,
                    default=20,
                    help="Number of epochs")
parser.add_argument("--size",
                    type=int,
                    default=256,
                    help="input image size")
parser.add_argument("--num_workers",
                    type=int,
                    default=0,
                    help="Number of workers for processing the data")
parser.add_argument("--classes",
                    type=int,
                    default=15,
                    help="Number of classes in the dataset")
parser.add_argument("--batch_size",
                    type=int,
                    default=8,
                    help="Number of batch size")
parser.add_argument("--lr",
                    type=float,
                    default=1e-3,
                    help="Number of learning rate")
parser.add_argument("--beta",
                    type=float,
                    default=0.999,
                    help="Param for weights - effective number")
parser.add_argument("--weight_decay",
                    type=float,
                    default=0.0001,
                    help="Number of weight decay")
parser.add_argument("--device_id",
                    type=str,
                    default="0",
                    help="GPU ID")
parser.add_argument("--titan",
                    action='store_true',
                    help="machine to run")
parser.add_argument("--limit",
                    type=int,
                    default=-1,
                    help="use to limit amount of data")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id

train_transform = transforms.Compose([
    transforms.Resize((args.size, args.size)),
    transforms.RandomHorizontalFlip()
])

valid_transform = transforms.Compose([
    transforms.Resize((args.size, args.size))
])

train_transform_albu = albu.Compose([
    albu.RandomResizedCrop(height=args.size, width=args.size, scale=(0.5, 1.0),
                           ratio=(0.8, 1.2), interpolation=1, p=1.0),
    albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, interpolation=1,
                          border_mode=0, value=0, p=0.25),
    albu.HorizontalFlip(p=0.5),
    albu.OneOf([
        albu.MotionBlur(p=.2),
        albu.MedianBlur(blur_limit=3, p=0.1),
        albu.Blur(blur_limit=3, p=0.1),
    ], p=0.25),
    albu.OneOf([
        albu.OpticalDistortion(p=0.3),
        albu.GridDistortion(p=0.3),
        albu.IAAPiecewiseAffine(p=0.1),
        albu.ElasticTransform(alpha=1, sigma=50, alpha_affine=40,
                              interpolation=1, border_mode=4,  # value=None, mask_value=None,
                              always_apply=False, approximate=False, p=0.3)
    ], p=0.3),
    albu.OneOf([
        albu.CLAHE(clip_limit=2),
        albu.IAASharpen(),
        albu.IAAEmboss(),
        albu.RandomBrightnessContrast(),
    ], p=0.25),
    albu.CoarseDropout(fill_value=0, p=0.25, max_height=32, max_width=32, max_holes=8),
    albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ToTensorV2(),
])

valid_transform_albu = albu.Compose([
    albu.Resize(args.size, args.size),
    albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ToTensorV2(),
])

train_dataset = NIHDataset(args.train_subset, args.labels, args.images, train_transform_albu, limit=args.limit)
train_indices = list(range(len(train_dataset)))
train_sampler = SubsetRandomSampler(train_indices)
train_loader = None

val_dataset = NIHDataset(args.test_subset, args.labels, args.images, valid_transform_albu, limit=args.limit)
val_indices = list(range(len(val_dataset)))
val_sampler = SubsetRandomSampler(val_indices)
validation_loader = None

if args.titan:

    print("running full dataset on titan")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               sampler=train_sampler,
                                               num_workers=args.num_workers)
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                                    sampler=val_sampler,
                                                    num_workers=args.num_workers)
else:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               num_workers=args.num_workers)
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                                    num_workers=args.num_workers)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# EFFNET
model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=args.classes, in_channels=args.in_channels)

# RESNET
# model = models.resnet34(pretrained=False)
# model.conv1 = nn.Conv1d(args.in_channels, 64, (7, 7), (2, 2), (3, 3), bias=False)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, args.classes)

model.cuda()

ens = get_ENS_weights(args.classes, list(sum(train_dataset.one_hot_labels)), beta=args.beta)
ens /= ens.max()
print(f"beta: {args.beta}, weights: {ens.tolist()}")
ens = ens.to(device=device, dtype=torch.float32)

criterion = nn.BCEWithLogitsLoss(weight=ens)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, min_lr=1e-6)

for epoch in range(args.epochs):
    start_time_epoch = time.time()
    print(f"Starting epoch {epoch + 1}")
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    auc_sum = 0.0
    preds = []
    labels = []

    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)
        optimizer.zero_grad()
        output = model(image)
        output = torch.sigmoid(output)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        pred = (output.data > 0.5).type(torch.float32)
        acc = accuracy_score(pred, label)
        preds.append(pred)
        labels.append(label)

        running_loss += loss.item()
        running_accuracy += acc

        if batch_idx % 10 == 0:
            print(" ", end="")
            print(f"Batch: {batch_idx + 1}/{len(train_loader)}"
                  f" Loss: {running_loss / ((batch_idx + 1)):.4f}"
                  f" Acc: {running_accuracy / (batch_idx + 1):.4f}"
                  f" Time: {time.time() - start_time_epoch:2f}")

    preds = torch.cat(preds, dim=0).tolist()
    labels = torch.cat(labels, dim=0).tolist()
    print("Training report:")
    print(classification_report(labels, preds, target_names=train_dataset.classes_names))

    val_running_loss = 0.0
    val_running_accuracy = 0.0
    val_preds = []
    val_labels = []
    model.eval()
    print("Validation: ")
    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(validation_loader):
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            output = model(image)
            output = torch.sigmoid(output)
            loss = criterion(output, label)
            pred = (output.data > 0.5).type(torch.float32)
            acc = accuracy_score(pred, label)

            val_preds.append(pred)
            val_labels.append(label)

            val_running_loss += loss.item()
            val_running_accuracy += acc

    train_loss = running_loss / len(train_loader)
    train_acc = running_accuracy / len(train_loader)
    val_loss = val_running_loss / len(validation_loader)
    val_acc = val_running_accuracy / len(validation_loader)

    scheduler.step(val_loss)

    for param_group in optimizer.param_groups:
        print(f"Current lr: {param_group['lr']}")

    print(f" Training Loss: {train_loss:.4f}"
          f" Training Acc: {train_acc:.4f}"
          f" Validation Loss: {val_loss:.4f}"
          f" Validation Acc: {val_acc:.4f}")

    val_preds = torch.cat(val_preds, dim=0).tolist()
    val_labels = torch.cat(val_labels, dim=0).tolist()
    print("Validation report:")
    print(classification_report(val_labels, val_preds, target_names=train_dataset.classes_names))
