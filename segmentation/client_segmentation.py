import logging
import os
import sys
import time
import warnings
from collections import OrderedDict

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler

from segmentation.data_loader import LungSegDataset
from segmentation.loss_functions import DiceLoss
from segmentation.models.unet import UNet

IMAGE_SIZE = 512
BATCH_SIZE = 16

hdlr = logging.StreamHandler()
logger = logging.getLogger(__name__)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    logger.info(f"CUDA is available: {device_name}")


# #############################################################################
# 1. PyTorch pipeline: model/train/test/dataloader
# #############################################################################

# Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def jaccard(outputs, targets):
    outputs = outputs.view(outputs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (outputs * targets).sum(1)
    union = (outputs + targets).sum(1) - intersection
    jac = (intersection + 0.001) / (union + 0.001)
    return jac.mean()


def train(net, train_loader, epochs):
    """Train the network on the training set."""
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=0.0001,
                                 weight_decay=0.0001
                                 )
    for epoch in range(epochs):
        start_time_epoch = time.time()
        logger.info('Starting epoch {}/{}'.format(epoch + 1, epochs))

        net.train()
        running_loss = 0.0
        running_jaccard = 0.0
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            outputs_masks = net(images)
            loss_seg = criterion(outputs_masks, masks)
            loss = loss_seg
            loss.backward()
            optimizer.step()

            jac = jaccard(outputs_masks.round(), masks)
            running_jaccard += jac.item()
            running_loss += loss.item()

            mask = masks[0, 0, :]
            out = outputs_masks[0, 0, :]
            res = torch.cat((mask, out), 1).cpu().detach()

            print('    ', end='')
            print('batch {:>3}/{:>3} loss: {:.4f}, Jaccard {:.4f}, learning time:  {:.2f}s\r' \
                  .format(batch_idx + 1, len(train_loader),
                          loss.item(), jac.item(),
                          time.time() - start_time_epoch))


def validate(net, val_loader):
    criterion = DiceLoss()

    """Validate the network on the entire validation set."""
    net.eval()
    val_running_loss = 0.0
    val_running_jac = 0.0
    for batch_idx, (images, masks) in enumerate(val_loader):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs_masks = net(images)
        loss_seg = criterion(outputs_masks, masks)
        loss = loss_seg

        val_running_loss += loss.item()
        jac = jaccard(outputs_masks.round(), masks)
        val_running_jac += jac.item()

        mask = masks[0, 0, :]
        out = outputs_masks[0, 0, :]
        res = torch.cat((mask, out), 1).cpu().detach()

    val_loss = val_running_loss / len(val_loader)
    val_jac = val_running_jac / len(val_loader)
    return val_loss, val_jac


def load_data(client_id, clients_number):
    """ Load Lung dataset for segmentation """

    scratch_path = os.environ['SCRATCH']
    masks_path = scratch_path + "/dataset/masks"
    images_path = scratch_path + "/dataset/images"

    dataset = LungSegDataset(client_id=client_id,
                             clients_number=clients_number,
                             path_to_images=images_path,
                             path_to_masks=masks_path,
                             image_size=IMAGE_SIZE)

    train_dataset = LungSegDataset(client_id=client_id,
                                   clients_number=clients_number,
                                   path_to_images=images_path,
                                   path_to_masks=masks_path,
                                   image_size=IMAGE_SIZE,
                                   mode="train")

    validation_dataset = LungSegDataset(client_id=client_id,
                                        clients_number=clients_number,
                                        path_to_images=images_path,
                                        path_to_masks=masks_path,
                                        image_size=IMAGE_SIZE,
                                        mode="valid")
    ids = np.array([i for i in range(len(dataset))])
    np.random.shuffle(ids)
    train_ids, val_ids = train_test_split(ids)

    train_sampler = SubsetRandomSampler(train_ids)
    val_sampler = SubsetRandomSampler(val_ids)

    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              sampler=train_sampler)

    val_loader = DataLoader(validation_dataset,
                            batch_size=BATCH_SIZE,
                            sampler=val_sampler)

    return train_loader, val_loader


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

def main():
    """Create model, load data, define Flower client, start Flower client."""
    arguments = sys.argv
    if len(arguments) < 4:
        raise TypeError("Client takes 3 arguments: server address, client id and clients number")

    server_addr = arguments[1]
    client_id = arguments[2]
    clients_number = arguments[3]
    # Load model
    net = UNet(input_channels=1,
               output_channels=64,
               n_classes=1).to(DEVICE)

    # Load data
    train_loader, val_loader = load_data(client_id, clients_number)

    # Flower client
    class SegmentationClient(fl.client.NumPyClient):
        def get_parameters(self):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            train(net, train_loader, epochs=1)
            return self.get_parameters(), len(train_loader), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = validate(net, val_loader)
            logger.info(f"Loss: {loss}, accuracy: {accuracy}")
            return float(loss), len(val_loader), {"accuracy": float(accuracy)}

    # Start client
    logger.info("Connecting to:", f"{server_addr}:8081")
    fl.client.start_numpy_client(f"{server_addr}:8081", client=SegmentationClient())


if __name__ == "__main__":
    main()
