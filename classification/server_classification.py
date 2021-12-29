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
from fl_covid_19_radiography_dataset import Covid19RDDataset
from utils import get_state_dict, get_test_transform_albu_NIH, test_NIH, parse_args, test_single_label, \
    get_test_transform_covid_19_rd, get_train_transform_covid_19_rd, test_single_label_patching
import torchvision
# from segmentation_models_pytorch import UnetPlusPlus


import sys

sys.path.append("..")
from segmentation.models.unet import UNet

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
    def __init__(self, args, segmentation_model=None):
        self.args = args
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(in_features=2048, out_features=args.classes)
        self.model = self.model.to(DEVICE)
        self.segmentation_model = segmentation_model

    def get_eval_fn(self, model, args, logger):
        test_transform = get_test_transform_covid_19_rd(args)
        test_dataset = RSNADataset(args, -1, args.clients_number, args.test_subset, args.images,
                                   transform=test_transform, debug=False, limit=args.limit,
                                   segmentation_model=self.segmentation_model)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                  num_workers=args.num_workers, pin_memory=True)

        classes_names = test_dataset.classes_names

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-5)

        def evaluate(weights):
            global ROUND
            state_dict = get_state_dict(model, weights)
            model.load_state_dict(state_dict, strict=True)
            if args.patches:
                test_acc, test_loss, report = test_single_label_patching(model, DEVICE, logger, test_dataset, criterion,
                                                                         optimizer, classes_names, args.k_patches_server)
            else:
                test_acc, test_loss, report = test_single_label(model, DEVICE, logger, test_loader, criterion,
                                                                optimizer, classes_names)
            torch.save(model.state_dict(), f'rsna_resnet_50_patching_1024-{ROUND}')
            loss.append(test_loss)
            acc.append(test_acc)
            reports.append(report)

            df = pd.DataFrame.from_dict(
                {'round': [i for i in range(ROUND + 1)], 'loss': loss, 'acc': acc, 'reports': reports})
            df.to_csv(f"rsna_resnet_50_r_patching_1024_{MAX_ROUNDS}-c_{CLIENTS}_bs_{BATCH_SIZE}_le_{LOCAL_EPOCHS}.csv")

            ROUND += 1
            return test_loss, {"test_acc": test_acc}

        return evaluate

    def get_strategy(self):
        return fl.server.strategy.FedAvg(
            fraction_fit=1,
            fraction_eval=1,
            min_fit_clients=CLIENTS,
            min_available_clients=CLIENTS,
            eval_fn=self.get_eval_fn(self.model, self.args, logger),
            initial_parameters=fl.common.weights_to_parameters([val.cpu().numpy() for _, val in self.model.state_dict().items()])
        )


class Covid19RDStrategyFactory:
    def __init__(self, args, segmentation_model=None):
        self.args = args
        self.model = torchvision.models.resnet34(pretrained=True)
        self.model.fc = torch.nn.Linear(in_features=512, out_features=args.classes)
        self.model.cuda()
        self.segmentation_model = segmentation_model

    def get_eval_fn(self, model, args, logger):
        test_transform = get_test_transform_covid_19_rd(args)
        test_dataset = Covid19RDDataset(args, -1, args.clients_number, args.test_subset, args.images,
                                        transform=test_transform, debug=False, limit=args.limit,
                                        segmentation_model=self.segmentation_model)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                  num_workers=args.num_workers)

        classes_names = test_dataset.classes_names

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=3e-5)

        def evaluate(weights):
            global ROUND
            state_dict = get_state_dict(model, weights)
            model.load_state_dict(state_dict, strict=True)
            if args.patches:
                test_acc, test_loss, report = test_single_label_patching(model, DEVICE, logger, test_dataset, criterion,
                                                                         optimizer, classes_names, args.k_patches_server)
            else:
                test_acc, test_loss, report = test_single_label(model, DEVICE, logger, test_loader, criterion,
                                                                optimizer, classes_names)
            torch.save(model.state_dict(), f'resnet_18-{ROUND}')
            loss.append(test_loss)
            acc.append(test_acc)
            reports.append(report)

            df = pd.DataFrame.from_dict(
                {'round': [i for i in range(ROUND + 1)], 'loss': loss, 'acc': acc, 'reports': reports})
            df.to_csv(f"Covid19RD_r_{MAX_ROUNDS}-c_{CLIENTS}_bs_{BATCH_SIZE}_le_{LOCAL_EPOCHS}.csv")

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

    # segmentation_model = UNet(input_channels=1,
    #                           output_channels=64,
    #                           n_classes=1).to(DEVICE)
    segmentation_model = None
    # segmentation_model = UnetPlusPlus('resnet34', in_channels=1, classes=1, activation='sigmoid').to(DEVICE)
    #
    # segmentation_model.load_state_dict(torch.load(args.segmentation_model, map_location=torch.device('cpu')))

    # Define strategy
    if args.patches:
        factory = RSNAStrategyFactory(args, segmentation_model)
    else:
        factory = RSNAStrategyFactory(args)
    strategy = factory.get_strategy()

    server_addr = socket.gethostname()
    # Start server
    logger.info(f"Starting server on {server_addr}")
    fl.server.start_server(
        server_address=f"{server_addr}:8087",
        config={"num_rounds": MAX_ROUNDS},
        strategy=strategy,
    )
