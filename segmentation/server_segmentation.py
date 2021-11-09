import logging
import socket

import flwr as fl
import torch
from torch.utils.data import DataLoader

from segmentation.client_segmentation import IMAGE_SIZE
from segmentation.data_loader import LungSegDataset
from segmentation.models.unet import UNet

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2


def get_eval_fn(net):
    masks_path = f"/net/archive/groups/plggsano/fl_msc/segmentation/ChestX_COVID-main/dataset/masks"
    images_path = f"/net/archive/groups/plggsano/fl_msc/segmentation/ChestX_COVID-main/dataset/images"
    test_dataset = LungSegDataset(path_to_images=images_path,
                                  path_to_masks=masks_path,
                                  image_size=IMAGE_SIZE,
                                  mode="test")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # todo: update weights in the server's model and run evaluation
    def evaluate(weights):
        pass

    return evaluate


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    net = UNet(input_channels=1,
               output_channels=64,
               n_classes=1).to(DEVICE)

    hdlr = logging.StreamHandler()
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,
        fraction_eval=0.5,
        min_fit_clients=1,
        # todo: uncomment when evaluation function is implemented
        # eval_fn=get_eval_fn(net),
        min_available_clients=2
    )
    server_addr = socket.gethostname()
    # Start server
    logger.info(f"Starting server on {server_addr}")
    fl.server.start_server(
        server_address="[::]:8081",
        config={"num_rounds": 3},
        strategy=strategy,
    )
