import logging
import os
import time

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import classification_report

from fl_nih_dataset import NIHDataset
from fl_rsna_dataset import RSNADataset
from fl_covid_19_radiography_dataset import Covid19RDDataset

import torchvision

from fl_mnist_dataset import MNISTDataset
from utils import get_state_dict, get_train_transformation_albu_NIH, accuracy_score, test_NIH, test_single_label, \
    get_test_transform_albu_NIH, parse_args, accuracy, get_train_transform_covid_19_rd, \
    get_test_transform_covid_19_rd

import torch.nn.functional as F

import sys

sys.path.append("..")
from segmentation.models.unet import UNet

hdlr = logging.StreamHandler()
logger = logging.getLogger(__name__)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.info(f"Device: {device}")
logger.info("CUDA_VISIBLE_DEVICES =" + str(os.environ['CUDA_VISIBLE_DEVICES']))

args = parse_args()


# os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id


def train_NIH(model, train_loader, criterion, optimizer, classes_names, epochs=1):
    for epoch in range(epochs):
        start_time_epoch = time.time()
        logger.info(f"Starting epoch {epoch + 1}")
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        preds = []
        labels = []

        for batch_idx, (image, label) in enumerate(train_loader):
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            optimizer.zero_grad()
            output = model(image)
            output = torch.sigmoid(output)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            pred = (output.data > 0.5).type(torch.float32)
            acc = accuracy_score(pred, label)
            preds.append(pred)
            labels.append(label)

            running_loss += loss.item()
            running_accuracy += acc

            if batch_idx % 10 == 0:
                logger.info(f"Batch: {batch_idx + 1}/{len(train_loader)}"
                            f" Loss: {running_loss / ((batch_idx + 1)):.4f}"
                            f" Acc: {running_accuracy / (batch_idx + 1):.4f}"
                            f" Time: {time.time() - start_time_epoch:2f}")

        preds = torch.cat(preds, dim=0).tolist()
        labels = torch.cat(labels, dim=0).tolist()
        logger.info("Training report:")
        logger.info(classification_report(labels, preds, target_names=classes_names))

        train_loss = running_loss / len(train_loader)
        train_acc = running_accuracy / len(train_loader)

        logger.info(f" Training Loss: {train_loss:.4f}"
                    f" Training Acc: {train_acc:.4f}")


def train_single_label(model, train_loader, criterion, optimizer, classes_names, epochs=1, K=1):
    for epoch in range(epochs):
        start_time_epoch = time.time()
        logger.info(f"Starting epoch {epoch + 1} / {epochs}")
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        preds = []
        labels = []
        total_batch_idx = 0        

        for i in range(K):
            logger.info(f"Starting repetition {i + 1} / {K} ")
            for batch_idx, (images, batch_labels) in enumerate(train_loader):
                images = images.to(device=device, dtype=torch.float32)
                batch_labels = batch_labels.to(device=device)

                logits = model(images)
                loss = criterion(logits, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_accuracy += accuracy(logits, batch_labels)

                y_pred = F.softmax(logits, dim=1)
                top_p, top_class = y_pred.topk(1, dim=1)

                labels.append(batch_labels.view(*top_class.shape))
                preds.append(top_class)

                if batch_idx % 10 == 0:
                    logger.info(f"Batch: {batch_idx + 1}/{len(train_loader)}"
                                f" Loss: {running_loss / ((total_batch_idx + 1)):.4f}"
                                f" Acc: {running_accuracy / (total_batch_idx + 1):.4f}"
                                f" Time: {time.time() - start_time_epoch:2f}")
                total_batch_idx += 1
        preds = torch.cat(preds, dim=0).tolist()
        labels = torch.cat(labels, dim=0).tolist()
        logger.info("Training report:")
        logger.info(classification_report(labels, preds, target_names=classes_names))

        train_loss = running_loss / total_batch_idx
        train_acc = running_accuracy / total_batch_idx

        logger.info(f" Training Loss: {train_loss:.4f}"
                    f" Training Acc: {train_acc:.4f}")


def load_data_NIH(client_id, clients_number):
    train_transform_albu = get_train_transformation_albu_NIH(args.size, args.size)
    test_transform_albu = get_test_transform_albu_NIH(args.size, args.size)

    if args.dataset == "chest":
        train_dataset = NIHDataset(client_id, clients_number, args.train_subset, args.labels, args.images,
                                   transform=train_transform_albu, limit=args.limit)
        test_dataset = NIHDataset(client_id, clients_number, args.test_subset, args.labels, args.images,
                                  transform=test_transform_albu, limit=args.limit)
    else:
        train_dataset = MNISTDataset(client_id, clients_number, args.train_subset, args.images,
                                     transform=train_transform_albu, limit=args.limit)
        test_dataset = MNISTDataset(client_id, clients_number, args.test_subset, args.images,
                                    transform=test_transform_albu, limit=args.limit)

    one_hot_labels = train_dataset.one_hot_labels
    classes_names = train_dataset.classes_names

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              num_workers=args.num_workers)
    return train_loader, test_loader, one_hot_labels, classes_names


def load_data_RSNA(client_id, clients_number):
    segmentation_model = None
    # if args.patches:
    #     segmentation_model = UNet(input_channels=1,
    #                               output_channels=64,
    #                               n_classes=1).to(device)
    #     segmentation_model.load_state_dict(torch.load(args.segmentation_model, map_location=torch.device('cpu')))

    train_transform = get_train_transform_covid_19_rd(args)
    train_dataset = RSNADataset(args, client_id, clients_number, args.train_subset, args.images,
                                transform=train_transform, debug=False, limit=args.limit,
                                segmentation_model=segmentation_model)

    test_transform = get_test_transform_covid_19_rd(args)
    test_dataset = RSNADataset(args, client_id, clients_number, args.test_subset, args.images,
                               transform=test_transform, debug=False, limit=args.limit,
                               segmentation_model=segmentation_model)

    classes_names = train_dataset.classes_names

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              num_workers=args.num_workers)
    return train_loader, test_loader, classes_names


def load_data_Covid19RD(client_id, clients_number):
    segmentation_model = None
    if args.patches:
        segmentation_model = UNet(input_channels=1,
                                  output_channels=64,
                                  n_classes=1).to(device)
        segmentation_model.load_state_dict(torch.load(args.segmentation_model, map_location=torch.device('cpu')))

    train_transform = get_train_transform_covid_19_rd(args)
    train_dataset = Covid19RDDataset(args, client_id, clients_number, args.train_subset, args.images,
                                     transform=train_transform, debug=False, limit=args.limit,
                                     segmentation_model=segmentation_model)

    test_transform = get_test_transform_covid_19_rd(args)
    test_dataset = Covid19RDDataset(args, client_id, clients_number, args.test_subset, args.images,
                                    transform=test_transform, debug=False, limit=args.limit,
                                    segmentation_model=segmentation_model)

    classes_names = train_dataset.classes_names

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              num_workers=args.num_workers)
    return train_loader, test_loader, classes_names


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

class ClassificationNIHClient(fl.client.NumPyClient):
    def __init__(self, client_id, clients_number):
        # Load model
        # EFFNET
        self.model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=args.classes,
                                                  in_channels=args.in_channels)
        self.model.cuda()

        # Load data
        self.train_loader, self.test_loader, self.one_hot_labels, self.classes_names = load_data_NIH(client_id,
                                                                                                     clients_number)

        self.criterion = nn.BCELoss(reduction='sum').to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, min_lr=1e-6)

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        logger.info("Loading parameters...")
        state_dict = get_state_dict(self.model, parameters)
        self.model.load_state_dict(state_dict, strict=True)
        logger.info("Parameters loaded")

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_NIH(self.model, self.train_loader, self.criterion, self.optimizer, self.classes_names,
                  epochs=args.local_epochs)
        return self.get_parameters(), len(self.train_loader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy, report = test_NIH(self.model, self.test_loader, device, logger, self.criterion, self.optimizer,
                                          self.scheduler, self.classes_names)
        logger.info(f"Loss: {loss}, accuracy: {accuracy}")
        logger.info(report)
        return float(loss), len(self.test_loader), {"accuracy": float(accuracy), "loss": float(loss)}


class ClassificationRSNAClient(fl.client.NumPyClient):
    def __init__(self, client_id, clients_number):
        # Load model
        self.model = torchvision.models.resnet34(pretrained=True)
        self.model.fc = torch.nn.Linear(in_features=512, out_features=args.classes)
        self.model.cuda()

        # Load data
        self.train_loader, self.test_loader, self.classes_names = load_data_RSNA(client_id, clients_number)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-5)

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        logger.info("Loading parameters...")
        state_dict = get_state_dict(self.model, parameters)
        self.model.load_state_dict(state_dict, strict=True)
        logger.info("Parameters loaded")

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_single_label(self.model, self.train_loader, self.criterion, self.optimizer, self.classes_names,
                           epochs=args.local_epochs, K=args.k_patches_client)
        return self.get_parameters(), len(self.train_loader), {}

    def evaluate(self, parameters, config):
        # TODO this is not reported at the moment
        self.set_parameters(parameters)
        loss, accuracy, report = test_single_label(self.model, self.test_loader, device, logger, self.criterion,
                                                   self.optimizer,
                                                   self.classes_names)
        logger.info(f"Loss: {loss}, accuracy: {accuracy}")
        logger.info(report)
        return float(loss), len(self.test_loader), {"accuracy": float(accuracy), "loss": float(loss)}


class ClassificationCovid19RDClient(fl.client.NumPyClient):
    def __init__(self, client_id, clients_number, args):
        # Load model
        self.model = torchvision.models.resnet34(pretrained=True)
        self.model.fc = torch.nn.Linear(in_features=512, out_features=args.classes)
        self.model.cuda()

        # Load data
        self.train_loader, self.test_loader, self.classes_names = load_data_Covid19RD(client_id, clients_number)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-5)

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        logger.info("Loading parameters...")
        state_dict = get_state_dict(self.model, parameters)
        self.model.load_state_dict(state_dict, strict=True)
        logger.info("Parameters loaded")

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_single_label(self.model, self.train_loader, self.criterion, self.optimizer, self.classes_names,
                           epochs=args.local_epochs, K=args.k_patches_client)
        return self.get_parameters(), len(self.train_loader), {}

    def evaluate(self, parameters, config):
        # TODO this is not reported at the moment
        self.set_parameters(parameters)
        loss, accuracy, report = test_single_label(self.model, self.test_loader, device, logger, self.criterion,
                                                   self.optimizer,
                                                   self.classes_names)
        logger.info(f"Loss: {loss}, accuracy: {accuracy}")
        logger.info(report)
        return float(loss), len(self.test_loader), {"accuracy": float(accuracy), "loss": float(loss)}


def main():
    """Create model, load data, define Flower client, start Flower client."""

    server_addr = args.node_name
    client_id = args.client_id
    clients_number = args.clients_number

    # Start client
    logger.info("Connecting to:" + f"{server_addr}:8087")
    fl.client.start_numpy_client(f"{server_addr}:8087",
                                 client=ClassificationRSNAClient(client_id, clients_number))


if __name__ == "__main__":
    main()
