import logging
import sys
import time
import warnings
from opacus import PrivacyEngine

import flwr as fl
import numpy as np
import torch
from segmentation_models_pytorch import UnetPlusPlus
from torch.utils.data import DataLoader, SubsetRandomSampler

from segmentation.common import get_state_dict, jaccard, validate, get_data_paths, get_model
from segmentation.data_loader import LungSegDataset
from segmentation.loss_functions import DiceLoss, DiceBCELoss

train_loader = None
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


def train(net, train_loader, epochs, lr, dice_only, optimizer_name, privacy_engine: PrivacyEngine = None):
    """Train the network on the training set."""

    criterion = get_criterion(dice_only)
    optimizer = get_optimizer(lr, net, optimizer_name)
    if privacy_engine:
        logger.info("Differential Privacy applied!")
        try:
            privacy_engine.attach(optimizer=optimizer)
        except Exception as e:
            logger.info(e)

    for epoch in range(epochs):
        start_time_epoch = time.time()
        logger.info('Starting epoch {}/{}'.format(epoch + 1, epochs))

        net.train()
        running_loss = 0.0
        running_jaccard = 0.0
        i = 0
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

            if i % 100 == 0:
                logger.info('batch {:>3}/{:>3} loss: {:.4f}, Jaccard {:.4f}, learning time:  {:.2f}s\r' \
                            .format(batch_idx + 1, len(train_loader),
                                    loss.item(), jac.item(),
                                    time.time() - start_time_epoch))
            i += 1

    epsilon, _ = optimizer.privacy_engine.get_privacy_spent()
    logger.info(f"epsilon = {epsilon:.2f}")


def get_optimizer(lr, net, optimizer_name):
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0001)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    elif optimizer_name == 'Adagrad':
        optimizer = torch.optim.Adagrad(net.parameters(), lr=lr)
    return optimizer


def get_criterion(dice_only):
    if dice_only:
        criterion = DiceLoss()
    else:
        criterion = DiceBCELoss()
    return criterion


def load_data(client_id, clients_number, batch_size, image_size):
    """ Load Lung dataset for segmentation """
    masks_path, images_path, labels = get_data_paths()

    dataset = LungSegDataset(client_id=client_id,
                             clients_number=clients_number,
                             path_to_images=images_path,
                             path_to_masks=masks_path,
                             image_size=image_size,
                             mode="train",
                             labels=labels)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)


def main():
    arguments = sys.argv
    if len(arguments) < 4:
        raise TypeError("Client takes 3 arguments: server address, client id and clients number")

    server_addr = arguments[1]
    client_id = int(arguments[2])
    clients_number = int(arguments[3])

    # Load model
    net = get_model().to(DEVICE)

    # Flower client
    class SegmentationClient(fl.client.NumPyClient):
        def __init__(self, net):
            self.privacy_engine = None

        def get_parameters(self):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            state_dict = get_state_dict(net, parameters)
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            global train_loader
            self.set_parameters(parameters)
            batch_size: int = config["batch_size"]
            image_size: int = config["image_size"]
            epochs: int = config["local_epochs"]
            lr: float = config["learning_rate"]
            optimizer_name: str = config["optimizer_name"]
            dice_only = config["dice_only"]
            if not train_loader:
                train_loader = load_data(client_id, clients_number, batch_size, image_size)
            if not self.privacy_engine:
                self.privacy_engine = PrivacyEngine(net,
                                                    batch_size=batch_size,
                                                    sample_size=len(train_loader),
                                                    max_grad_norm=1.0,
                                                    noise_multiplier=1.0)

            try:
                train(net, train_loader, epochs=epochs, lr=lr, dice_only=dice_only,
                      optimizer_name=optimizer_name, privacy_engine=self.privacy_engine)
            except Exception as e:
                logger.info(e)
            return self.get_parameters(), len(train_loader), {}

        def evaluate(self, parameters, config):
            pass
            # self.set_parameters(parameters)
            # loss, jaccard_score = validate(net, val_loader, DEVICE)
            # logger.info(f"Loss: {loss}, jaccard score: {jaccard_score}")
            # return float(loss), len(val_loader), {"jaccard_score": float(jaccard_score)}

    # Start client
    logger.info(f"Connecting to: {server_addr}:8081")
    while True:
        try:
            client = SegmentationClient(net)
            fl.client.start_numpy_client(f"{server_addr}:8081", client=client)
        except Exception as e:
            logger.info("Exception occurred")
            logger.info(e)
            time.sleep(120)


if __name__ == "__main__":
    main()
