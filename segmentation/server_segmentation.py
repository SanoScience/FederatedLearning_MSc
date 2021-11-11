import logging
import socket

import flwr as fl
from torch.utils.data import DataLoader

from segmentation.common import *
from segmentation.client_segmentation import IMAGE_SIZE
from segmentation.data_loader import LungSegDataset
from segmentation.models.unet import UNet

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
ROUND = 0


def get_eval_fn(net):
    masks_path, images_path = get_data_paths()
    test_dataset = LungSegDataset(path_to_images=images_path,
                                  path_to_masks=masks_path,
                                  image_size=IMAGE_SIZE,
                                  mode="test")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # todo: update weights in the server's model and run evaluation
    def evaluate(weights):
        global ROUND
        state_dict = get_state_dict(net, weights)
        net.load_state_dict(state_dict, strict=True)
        val_loss, val_jacc = validate(net, test_loader, DEVICE)
        torch.save(net.state_dict(), f'unet_{ROUND}')
        ROUND += 1
        return val_loss, {"val_jacc": val_jacc}

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
        fraction_fit=0.75,
        fraction_eval=0.75,
        min_fit_clients=1,
        min_eval_clients=1,
        eval_fn=get_eval_fn(net),
        min_available_clients=2
    )
    server_addr = socket.gethostname()
    # Start server
    logger.info(f"Starting server on {server_addr}")
    fl.server.start_server(
        server_address=f"{server_addr}:8081",
        config={"num_rounds": 3},
        strategy=strategy,
    )
