import flwr as fl
import socket
import logging
from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from fl_mnist_dataset import MNISTDataset
from fl_nih_dataset import NIHDataset
from fl_rsna_dataset import RSNADataset
from utils import get_state_dict, get_test_transform_albu_NIH, test_NIH, parse_args, test_RSNA
import timm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ROUND = 0
MAX_ROUNDS = 0
BATCH_SIZE = 0
CLIENTS = 0
LOCAL_EPOCHS = 0

loss = []
acc = []
reports = []


class NIHStrategyFactory:
    def __int__(self, args):
        self.args = args
        self.model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=args.classes,
                                                  in_channels=args.in_channels)
        self.model.cuda()

    def get_eval_fn(self, model, args, logger):
        test_transform_albu = get_test_transform_albu_NIH(args.size, args.size)
        if args.dataset == "chest":
            test_dataset = NIHDataset(-1, args.clients_number, args.test_subset, args.labels, args.images,
                                      transform=test_transform_albu, limit=args.limit)
        else:
            test_dataset = MNISTDataset(-1, args.clients_number, args.test_subset, args.images,
                                        transform=test_transform_albu, limit=args.limit)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                  num_workers=args.num_workers)

        classes_names = test_dataset.classes_names

        criterion = nn.BCELoss(reduction='sum').to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, min_lr=1e-6)

        def evaluate(weights):
            global ROUND
            state_dict = get_state_dict(model, weights)
            model.load_state_dict(state_dict, strict=True)
            test_acc, test_loss, report = test_NIH(model, DEVICE, logger, test_loader, criterion, optimizer, scheduler,
                                                   classes_names)
            torch.save(model.state_dict(), f'efficientnet-b4_{ROUND}')
            loss.append(test_loss)
            acc.append(test_acc)
            reports.append(report)

            df = pd.DataFrame.from_dict(
                {'round': [i for i in range(ROUND + 1)], 'loss': loss, 'acc': acc, 'reports': reports})
            df.to_csv(f"nih_r_{MAX_ROUNDS}-c_{CLIENTS}_bs_{BATCH_SIZE}_le_{LOCAL_EPOCHS}.csv")

            ROUND += 1
            return test_loss, {"test_acc": test_acc}

        return evaluate

    def get_strategy(self):
        return fl.server.strategy.FedAvg(
            fraction_fit=1,
            fraction_eval=1,
            min_fit_clients=4,
            min_available_clients=CLIENTS,
            eval_fn=self.get_eval_fn(self.model, self.args, logger),
            initial_parameters=[val.cpu().numpy() for _, val in self.model.state_dict().items()]
        )


class RSNAStrategyFactory:
    def __int__(self, args):
        self.args = args
        # EFFNET
        self.model = timm.create_model('tf_efficientnet_b4_ns', pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=1792, out_features=625),  # 1792 is the original in_features
            nn.ReLU(),  # ReLu to be the activation function
            nn.Dropout(p=0.3),
            nn.Linear(in_features=625, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=2),
        )
        self.model.to(DEVICE)

    def get_eval_fn(self, model, args, logger):
        test_dataset = RSNADataset(-1, args.clients_number, args.test_subset, args.size, args.images,
                                   is_training=False, debug=False, limit=args.limit)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                  num_workers=args.num_workers)

        classes_names = test_dataset.classes_names

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        def evaluate(weights):
            global ROUND
            state_dict = get_state_dict(model, weights)
            model.load_state_dict(state_dict, strict=True)
            test_acc, test_loss, report = test_RSNA(model, DEVICE, logger, test_loader, criterion, optimizer,
                                                   classes_names)
            torch.save(model.state_dict(), f'tf_efficientnet_b4_ns-{ROUND}')
            loss.append(test_loss)
            acc.append(test_acc)
            reports.append(report)

            df = pd.DataFrame.from_dict(
                {'round': [i for i in range(ROUND + 1)], 'loss': loss, 'acc': acc, 'reports': reports})
            df.to_csv(f"rsna_r_{MAX_ROUNDS}-c_{CLIENTS}_bs_{BATCH_SIZE}_le_{LOCAL_EPOCHS}.csv")

            ROUND += 1
            return test_loss, {"test_acc": test_acc}

        return evaluate

    def get_strategy(self):
        return fl.server.strategy.FedAvg(
            fraction_fit=1,
            fraction_eval=1,
            min_fit_clients=4,
            min_available_clients=CLIENTS,
            eval_fn=self.get_eval_fn(self.model, self.args, logger),
            initial_parameters=[val.cpu().numpy() for _, val in self.model.state_dict().items()]
        )


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    hdlr = logging.StreamHandler()
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    args = parse_args()

    MAX_ROUNDS = args.num_rounds
    CLIENTS = args.clients_number
    LOCAL_EPOCHS = args.local_epochs
    BATCH_SIZE = args.batch_size

    # Define strategy
    factory = RSNAStrategyFactory()
    strategy = factory.get_strategy()

    server_addr = socket.gethostname()
    # Start server
    logger.info(f"Starting server on {server_addr}")
    fl.server.start_server(
        server_address=f"{server_addr}:8081",
        config={"num_rounds": MAX_ROUNDS},
        strategy=strategy,
    )
