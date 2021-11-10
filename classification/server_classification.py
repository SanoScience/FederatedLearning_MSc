import flwr as fl
import socket
import logging

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    hdlr = logging.StreamHandler()
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,
        fraction_eval=0.5,
        min_fit_clients=1,
        min_available_clients=2
    )
    server_addr = socket.gethostname()
    # Start server
    logger.info(f"Starting server on {server_addr}")
    fl.server.start_server(
        server_address=f"{server_addr}:8081",
        config={"num_rounds": 10},
        strategy=strategy,
    )
