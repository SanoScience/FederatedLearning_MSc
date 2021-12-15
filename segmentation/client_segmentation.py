import logging
import sys
import time
import warnings

import flwr as fl
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler

from segmentation.common import get_state_dict, jaccard, validate, get_data_paths
from segmentation.data_loader import LungSegDataset
from segmentation.loss_functions import DiceLoss, DiceBCELoss
from segmentation.models.unet import UNet

IMAGE_SIZE = 1024
BATCH_SIZE = 2

hdlr = logging.StreamHandler()
logger = logging.getLogger(__name__)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)
torch.cuda.empty_cache()
warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    logger.info(f"CUDA is available: {device_name}")


def train(net, train_loader, epochs, lr, dice_only):
    """Train the network on the training set."""
    if dice_only:
        criterion = DiceLoss()
    else:
        criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=lr,
                                 weight_decay=0.0001)
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
            loss = criterion(outputs_masks, masks)
            loss.backward()
            optimizer.step()

            jac = jaccard(outputs_masks.round(), masks)
            running_jaccard += jac.item()
            running_loss += loss.item()

            # mask = masks[0, 0, :]
            # out = outputs_masks[0, 0, :]
            # res = torch.cat((mask, out), 1).cpu().detach()

            logger.info('batch {:>3}/{:>3} loss: {:.4f}, Jaccard {:.4f}, learning time:  {:.2f}s\r' \
                        .format(batch_idx + 1, len(train_loader),
                                loss.item(), jac.item(),
                                time.time() - start_time_epoch))


def load_data(client_id, clients_number):
    """ Load Lung dataset for segmentation """
    masks_path, images_path = get_data_paths()

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

    # ids = np.array([i for i in range(len(dataset))])
    # np.random.shuffle(ids)
    # train_ids, val_ids = train_test_split(ids)
    # logger.info(f"Dataset size: {len(train_dataset)}; {len(ids)} ")
    # train_sampler = SubsetRandomSampler(train_ids)
    # val_sampler = SubsetRandomSampler(val_ids)

    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,)
                              # sampler=train_sampler)

    # val_loader = DataLoader(validation_dataset,
    #                         batch_size=BATCH_SIZE,
    #                         sampler=val_sampler)

    return train_loader, None


def main():
    arguments = sys.argv
    if len(arguments) < 4:
        raise TypeError("Client takes 3 arguments: server address, client id and clients number")

    server_addr = arguments[1]
    client_id = int(arguments[2])
    clients_number = int(arguments[3])

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
            state_dict = get_state_dict(net, parameters)
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            # todo: use if necessary :)
            # batch_size: int = config["batch_size"]
            epochs: int = config["local_epochs"]
            lr: int = config["learning_rate"]
            dice_only = config["dice_only"]
            train(net, train_loader, epochs=epochs, lr=lr, dice_only=dice_only)
            return self.get_parameters(), len(train_loader), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, jaccard_score = validate(net, val_loader, DEVICE)
            logger.info(f"Loss: {loss}, jaccard score: {jaccard_score}")
            return float(loss), len(val_loader), {"jaccard_score": float(jaccard_score)}

    # Start client
    logger.info(f"Connecting to: {server_addr}:8081")
    fl.client.start_numpy_client(f"{server_addr}:8081", client=SegmentationClient())


if __name__ == "__main__":
    main()
