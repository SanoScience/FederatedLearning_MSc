import logging
import socket

import flwr as fl
from torch.utils.data import DataLoader
import pandas as pd
from segmentation.common import *
from segmentation.client_segmentation import IMAGE_SIZE
from segmentation.data_loader import LungSegDataset
from segmentation.models.unet import UNet

loss = []
jacc = []
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
ROUND = 0
MAX_ROUND = 5
CLIENTS = 3
LOCAL_EPOCHS = 1


def fit_config(rnd: int):
    config = {
        "batch_size": BATCH_SIZE,
        "local_epochs": LOCAL_EPOCHS,
    }
    return config


def get_eval_fn(net):
    masks_path, images_path = get_data_paths()
    test_dataset = LungSegDataset(path_to_images=images_path,
                                  path_to_masks=masks_path,
                                  image_size=IMAGE_SIZE,
                                  mode="test")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # todo: update weights in the server's model and run evaluation
    def evaluate(weights):
        global ROUND, MAX_ROUND
        state_dict = get_state_dict(net, weights)
        net.load_state_dict(state_dict, strict=True)
        val_loss, val_jacc = validate(net, test_loader, DEVICE)
        torch.save(net.state_dict(), f'unet_{ROUND}')
        loss.append(val_loss)
        jacc.append(jacc)
        if MAX_ROUND == ROUND:
            df = pd.DataFrame.from_dict({'round': [i for i in range(MAX_ROUND)], 'loss': loss, 'jaccard': jacc})
            df.to_csv(f"r_{MAX_ROUND}-c_{CLIENTS}_bs_{BATCH_SIZE}_le_{LOCAL_EPOCHS}.csv")
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
        min_available_clients=CLIENTS,
        on_fit_config_fn=fit_config,
        initial_parameters=[val.cpu().numpy() for _, val in net.state_dict().items()]
    )
    server_addr = socket.gethostname()
    # Start server
    logger.info(f"Starting server on {server_addr}")
    fl.server.start_server(
        server_address=f"{server_addr}:8081",
        config={"num_rounds": MAX_ROUND},
        strategy=strategy,
    )
