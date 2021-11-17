import os
import torch
from collections import OrderedDict
from sklearn.metrics import classification_report
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np
import argparse

DATASET_PATH_BASE = os.path.expandvars("$SCRATCH/fl_msc/classification/NIH/data/")

def accuracy_score(pred, actual):
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


def get_state_dict(model, parameters):
    params_dict = []
    for i, k in enumerate(list(model.state_dict().keys())):
        p = parameters[i]
        if 'num_batches_tracked' in k:
            p = p.reshape(p.size)
        params_dict.append((k, p))
    return OrderedDict({k: torch.Tensor(v) for k, v in params_dict})


def get_train_transformation_albu(height, width):
    return albu.Compose([
        albu.RandomResizedCrop(height=height, width=width, scale=(0.5, 1.0),
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


def get_test_transform_albu(height, width):
    return albu.Compose([
        albu.Resize(height, width),
        albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ToTensorV2(),
    ])


def test(model, device, logger, test_loader, criterion, optimizer, scheduler, classes_names):
    test_running_loss = 0.0
    test_running_accuracy = 0.0
    test_preds = []
    test_labels = []
    model.eval()
    logger.info("Testing: ")
    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(test_loader):
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            output = model(image)
            output = torch.sigmoid(output)
            loss = criterion(output, label)
            pred = (output.data > 0.5).type(torch.float32)
            acc = accuracy_score(pred, label)

            test_preds.append(pred)
            test_labels.append(label)

            test_running_loss += loss.item()
            test_running_accuracy += acc
            if batch_idx % 50 == 0:
                logger.info(f"batch_idx: {batch_idx}")

    test_loss = test_running_loss / len(test_loader)
    test_acc = test_running_accuracy / len(test_loader)

    scheduler.step(test_loss)

    for param_group in optimizer.param_groups:
        logger.info(f"Current lr: {param_group['lr']}")

    logger.info(f" Test Loss: {test_loss:.4f}"
                f" Test Acc: {test_acc:.4f}")

    test_preds = torch.cat(test_preds, dim=0).tolist()
    test_labels = torch.cat(test_labels, dim=0).tolist()
    logger.info("Test report:")
    report = classification_report(test_labels, test_preds, target_names=classes_names)
    logger.info(report)
    return test_acc, test_loss, report


def parse_args():
    parser = argparse.ArgumentParser(description="Train classifier to detect covid on CXR images.")

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
                        help="Path to the file with training/validation dataset files list")
    parser.add_argument("--test_subset",
                        type=str,
                        default=os.path.join(DATASET_PATH_BASE, "partitions/nih_test_list.txt"),
                        help="Path to the file with test dataset files list")
    parser.add_argument("--in_channels",
                        type=int,
                        default=3,
                        help="Number of input channels")
    parser.add_argument("--local_epochs",
                        type=int,
                        default=3,
                        help="Number of local epochs")
    parser.add_argument("--size",
                        type=int,
                        default=512,
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

    parser.add_argument("--node_name",
                        type=str,
                        default="p001",
                        help="server node name")

    parser.add_argument("--client_id",
                        type=int,
                        default=0,
                        help="ID of the client")

    parser.add_argument("--clients_number",
                        type=int,
                        default=4,
                        help="number of the clients")

    parser.add_argument("--dataset",
                        type=str,
                        default="chest",
                        help="kind of dataset: chest/mnist")

    parser.add_argument("--num_rounds",
                        type=int,
                        default=20,
                        help="number of rounds")

    args = parser.parse_args()
    return args
