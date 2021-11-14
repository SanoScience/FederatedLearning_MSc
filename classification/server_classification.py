import flwr as fl
import socket
import logging
import argparse
from efficientnet_pytorch import EfficientNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train classifier to detect covid on CXR images.")
    parser.add_argument("--classes",
                        type=int,
                        default=15,
                        help="Number of classes in the dataset")
    parser.add_argument("--in_channels",
                        type=int,
                        default=3,
                        help="Number of input channels")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    hdlr = logging.StreamHandler()
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    args = parse_args()

    model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=args.classes, in_channels=args.in_channels)
    model.cuda()

    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,
        fraction_eval=0.5,
        min_fit_clients=1,
        min_available_clients=2,
        initial_parameters=[val.cpu().numpy() for _, val in model.state_dict().items()]
    )
    server_addr = socket.gethostname()
    # Start server
    logger.info(f"Starting server on {server_addr}")
    fl.server.start_server(
        server_address=f"{server_addr}:8081",
        config={"num_rounds": 10},
        strategy=strategy,
    )
