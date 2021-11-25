import logging
import socket

import click
import flwr as fl
import pandas as pd
from torch.utils.data import DataLoader

from segmentation.client_segmentation import IMAGE_SIZE
from segmentation.common import *
from segmentation.data_loader import LungSegDataset
from segmentation.models.unet import UNet

loss = []
jacc = []
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
ROUND = 0
MAX_ROUND = 5
CLIENTS = 3
FED_AGGREGATION_STRATEGY = 'FedAvg'
LOCAL_EPOCHS = 1
MIN_FIT_CLIENTS = 2
FRACTION_FIT = 0.75

strategies = {'FedAdam': fl.server.strategy.FedAdam,
              'FedAvg': fl.server.strategy.FedAvg,
              'FedAdagrad': fl.server.strategy.FedAdagrad}


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

    def evaluate(weights):
        global ROUND, MAX_ROUND
        state_dict = get_state_dict(net, weights)
        net.load_state_dict(state_dict, strict=True)
        val_loss, val_jacc = validate(net, test_loader, DEVICE)
        torch.save(net.state_dict(),
                   f'unet_{ROUND}_jacc_{round(val_jacc, 3)}_loss_{round(val_loss, 3)}_agg_{FED_AGGREGATION_STRATEGY}')
        loss.append(val_loss)
        jacc.append(jacc)
        if MAX_ROUND == ROUND:
            df = pd.DataFrame.from_dict({'round': [i for i in range(MAX_ROUND + 1)], 'loss': loss, 'jaccard': jacc})
            df.to_csv(
                f"r_{MAX_ROUND}-c_{CLIENTS}_bs_{BATCH_SIZE}_le_{LOCAL_EPOCHS}_fs_{FED_AGGREGATION_STRATEGY}_mf_{MIN_FIT_CLIENTS}_ff_{FRACTION_FIT}.csv")
        ROUND += 1
        return val_loss, {"val_jacc": val_jacc, "val_dice_loss": val_loss}

    return evaluate


@click.command()
@click.option('--le', default=LOCAL_EPOCHS, type=int, help='Local epochs performed by clients')
@click.option('--a', default=FED_AGGREGATION_STRATEGY, type=str,
              help='Aggregation strategy (FedAvg, FedAdam, FedAdagrad')
@click.option('--c', default=CLIENTS, type=int, help='Clients number')
@click.option('--r', default=MAX_ROUND, type=int, help='Rounds of training')
@click.option('--mf', default=MIN_FIT_CLIENTS, type=int, help='Min fit clients')
@click.option('--ff', default=FRACTION_FIT, type=float, help='Fraction fit')
@click.option('--bs', default=BATCH_SIZE, type=int, help='Batch size')
def run_server(le, a, c, r, mf, ff, bs):
    global LOCAL_EPOCHS, FED_AGGREGATION_STRATEGY, CLIENTS, MAX_ROUND, MIN_FIT_CLIENTS, FRACTION_FIT, BATCH_SIZE
    LOCAL_EPOCHS = le
    FED_AGGREGATION_STRATEGY = a
    CLIENTS = c
    MAX_ROUND = r
    MIN_FIT_CLIENTS = mf
    FRACTION_FIT = ff
    BATCH_SIZE = bs

    # Initialize logger
    logger = logging.getLogger(__name__)
    hdlr = logging.StreamHandler()
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    logger.info("Parsing arguments")

    # Define model
    net = UNet(input_channels=1,
               output_channels=64,
               n_classes=1).to(DEVICE)

    # Define strategy
    strategy = strategies[FED_AGGREGATION_STRATEGY](
        fraction_fit=FRACTION_FIT,
        fraction_eval=0.75,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_eval_clients=2,
        eval_fn=get_eval_fn(net),
        min_available_clients=CLIENTS,
        on_fit_config_fn=fit_config,
        initial_parameters=fl.common.weights_to_parameters([val.cpu().numpy() for _, val in net.state_dict().items()]),
        eta=0.25
    )

    # Start server
    server_addr = socket.gethostname()
    logger.info(f"Starting server on {server_addr}")
    fl.server.start_server(
        server_address=f"{server_addr}:8081",
        config={"num_rounds": MAX_ROUND},
        strategy=strategy,
    )


if __name__ == "__main__":
    run_server()
