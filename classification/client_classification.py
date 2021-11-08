import logging
import os
import sys
import time
from collections import OrderedDict

import flwr as fl
import numpy as np
import torch
import torch
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import classification_report
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import argparse
from fl_nih_dataset import NIHDataset
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

hdlr = logging.StreamHandler()
logger = logging.getLogger(__name__)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
                    default=160,
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
                    default=1,
                    help="number of the clients")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id


def train(model, train_loader, criterion, optimizer, classes_names, epochs=1):
    for epoch in range(epochs):
        start_time_epoch = time.time()
        print(f"Starting epoch {epoch + 1}")
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
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
        print(classification_report(labels, preds, target_names=classes_names))

        train_loss = running_loss / len(train_loader)
        train_acc = running_accuracy / len(train_loader)

        print(f" Training Loss: {train_loss:.4f}"
              f" Training Acc: {train_acc:.4f}")


def validate(model, validation_loader, criterion, optimizer, scheduler, classes_names):
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

    val_loss = val_running_loss / len(validation_loader)
    val_acc = val_running_accuracy / len(validation_loader)

    scheduler.step(val_loss)

    for param_group in optimizer.param_groups:
        print(f"Current lr: {param_group['lr']}")

    print(f" Validation Loss: {val_loss:.4f}"
          f" Validation Acc: {val_acc:.4f}")

    val_preds = torch.cat(val_preds, dim=0).tolist()
    val_labels = torch.cat(val_labels, dim=0).tolist()
    print("Validation report:")
    print(classification_report(val_preds, val_labels, target_names=classes_names))
    return val_acc, val_loss


def get_train_transformation_albu():
    return albu.Compose([
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


def get_valid_transform_albu():
    return albu.Compose([
        albu.Resize(args.size, args.size),
        albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ToTensorV2(),
    ])


def load_data(client_id, clients_number):
    train_transform_albu = get_train_transformation_albu()
    valid_transform_albu = get_valid_transform_albu()
    train_dataset = NIHDataset(args.train_subset, args.labels, args.images, train_transform_albu, limit=args.limit)
    val_dataset = NIHDataset(args.test_subset, args.labels, args.images, valid_transform_albu, limit=args.limit)

    classes_names = train_dataset.classes_names

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               num_workers=args.num_workers)
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                                    num_workers=args.num_workers)
    return train_loader, validation_loader, classes_names


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

def main():
    """Create model, load data, define Flower client, start Flower client."""

    server_addr = args.node_name
    client_id = args.client_id
    clients_number = args.clients_number
    # Load model
    # EFFNET
    model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=args.classes, in_channels=args.in_channels)
    model.cuda()

    # Load data
    train_loader, validation_loader, classes_names = load_data(client_id, clients_number)

    ens = get_ENS_weights(args.classes, list(sum(classes_names)), beta=args.beta)
    ens /= ens.max()
    print(f"beta: {args.beta}, weights: {ens.tolist()}")
    ens = ens.to(device=device, dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(weight=ens)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, min_lr=1e-6)

    # Flower client
    class ClassificationClient(fl.client.NumPyClient):
        def get_parameters(self):
            return [val.cpu().numpy() for _, val in model.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            train(model, train_loader, criterion, optimizer, classes_names, epochs=args.epochs)
            return self.get_parameters(), len(train_loader), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = validate(model, validation_loader, criterion, optimizer, scheduler, classes_names)
            logger.info(f"Loss: {loss}, accuracy: {accuracy}")
            return float(loss), len(validation_loader), {"accuracy": float(accuracy)}

    # Start client
    logger.info("Connecting to:", f"{server_addr}:8081")
    fl.client.start_numpy_client(f"{server_addr}:8081", client=ClassificationClient())


if __name__ == "__main__":
    main()
