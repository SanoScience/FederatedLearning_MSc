import flwr as fl
import socket
import logging
from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from classification.fl_mnist_dataset import MNISTDataset
from classification.fl_nih_dataset import NIHDataset
from classification.utils import get_state_dict, get_test_transform_albu, get_ENS_weights, test, parse_args

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ROUND = 0
MAX_ROUNDS = 0
BATCH_SIZE = 8
MAX_ROUND = 5
CLIENTS = 3
LOCAL_EPOCHS = 1

loss = []
acc = []
reports = []


def get_eval_fn(model, args, logger):
    test_transform_albu = get_test_transform_albu(args.size, args.size)
    if args.dataset == "chest":
        test_dataset = NIHDataset(-1, args.clients_number, args.test_subset, args.labels, args.images,
                                  transform=test_transform_albu, limit=args.limit)
    else:
        test_dataset = MNISTDataset(-1, args.clients_number, args.test_subset, args.images)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              num_workers=args.num_workers)

    one_hot_labels = test_dataset.one_hot_labels
    classes_names = test_dataset.classes_names

    ens = get_ENS_weights(args.classes, list(sum(one_hot_labels)), beta=args.beta)
    ens /= ens.max()
    logger.info(f"beta: {args.beta}, weights: {ens.tolist()}")
    ens = ens.to(device=DEVICE, dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(weight=ens)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, min_lr=1e-6)

    def evaluate(weights):
        global ROUND, MAX_ROUNDS
        state_dict = get_state_dict(model, weights)
        model.load_state_dict(state_dict, strict=True)
        test_acc, test_loss, report = test(model, DEVICE, logger, test_loader, criterion, optimizer, scheduler,
                                           classes_names)
        torch.save(model.state_dict(), f'efficientnet-b7_{ROUND}')
        loss.append(test_loss)
        acc.append(test_acc)
        reports.append(report)

        df = pd.DataFrame.from_dict(
            {'round': [i for i in range(MAX_ROUND + 1)], 'loss': loss, 'acc': acc, 'reports': reports})
        df.to_csv(f"r_{MAX_ROUND}-c_{CLIENTS}_bs_{BATCH_SIZE}_le_{LOCAL_EPOCHS}.csv")

        ROUND += 1
        return test_loss, {"test_acc": test_acc}

    return evaluate


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    hdlr = logging.StreamHandler()
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    args = parse_args()

    model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=args.classes, in_channels=args.in_channels)
    model.cuda()

    MAX_ROUNDS = args.num_rounds

    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.75,
        fraction_eval=0.75,
        min_fit_clients=1,
        min_available_clients=CLIENTS,
        eval_fn=get_eval_fn(model, args, logger),
        initial_parameters=[val.cpu().numpy() for _, val in model.state_dict().items()]
    )
    server_addr = socket.gethostname()
    # Start server
    logger.info(f"Starting server on {server_addr}")
    fl.server.start_server(
        server_address=f"{server_addr}:8081",
        config={"num_rounds": MAX_ROUNDS},
        strategy=strategy,
    )
