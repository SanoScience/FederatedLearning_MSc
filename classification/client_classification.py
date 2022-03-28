import os
import logging
import time

import flwr as fl
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToDevice, ToTorchImage, Cutout, NormalizeImage, Convert, ToTensor
from ffcv.transforms.common import Squeeze
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder, NDArrayDecoder

from data_selector import IIDSelector

from utils import get_state_dict, accuracy, get_model, get_data_paths, get_beton_data_paths, \
    get_type_of_dataset, get_class_names, log_gpu_utilization_csv, make_round_gpu_metrics_dir, save_round_gpu_csv

import torch.nn.functional as F
import click
import pandas as pd

IMAGE_SIZE = 224
LIMIT = -1
CLIENT_ID = 0
D_NAME = 'nih'
SERVER_ADDRESS = ''
ROUND = 0
HPC_LOG = True

hdlr = logging.StreamHandler()
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(hdlr)
LOGGER.setLevel(logging.INFO)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_single_label(model, train_loader, criterion, optimizer, classes_names, epochs):
    gpu_stats_dfs = []
    if HPC_LOG:
        make_round_gpu_metrics_dir(SERVER_ADDRESS, D_NAME, CLIENT_ID, ROUND)
    for epoch in range(epochs):
        start_time_epoch = time.time()
        LOGGER.info(f"Starting epoch {epoch + 1} / {epochs}")
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        labels = torch.IntTensor().to(device)
        preds = torch.IntTensor().to(device)

        for batch_idx, (images, batch_labels) in enumerate(train_loader):
            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, batch_labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_accuracy += accuracy(logits, batch_labels)

            y_pred = F.softmax(logits, dim=1)
            _, top_class = y_pred.topk(1, dim=1)

            preds = torch.cat((preds, top_class.data), 0)
            labels = torch.cat((labels, batch_labels.data), 0)

            if batch_idx % 10 == 0:
                LOGGER.info(f"Batch: {batch_idx + 1}/{len(train_loader)}"
                            f" Loss: {running_loss / (batch_idx + 1):.4f}"
                            f" Acc: {running_accuracy / (batch_idx + 1):.4f}"
                            f" Time: {time.time() - start_time_epoch:2f}")
                if HPC_LOG and batch_idx % 300 == 0:
                    gpu_stats_dfs.append(log_gpu_utilization_csv(D_NAME, CLIENT_ID, ROUND, epoch, batch_idx))
        preds = preds.cpu().numpy().astype(np.int32)
        labels = labels.cpu().numpy().astype(np.int32)
        LOGGER.info("Training report:")
        LOGGER.info(classification_report(labels, preds, target_names=classes_names))

        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(labels, preds)

        LOGGER.info(f" Training Loss: {train_loss:.4f}"
                    f" Training Acc: {train_acc:.4f}")
        if HPC_LOG:
            save_round_gpu_csv(gpu_stats_dfs, SERVER_ADDRESS, D_NAME, str(CLIENT_ID), ROUND)


def train_multi_label(model, train_loader, criterion, optimizer, classes_names, epochs):
    gpu_stats_dfs = []
    if HPC_LOG:
        make_round_gpu_metrics_dir(SERVER_ADDRESS, D_NAME, CLIENT_ID, ROUND)
    for epoch in range(epochs):
        start_time_epoch = time.time()
        LOGGER.info(f"Starting epoch {epoch + 1} / {epochs}")
        model.train()
        running_loss = 0.0
        labels = torch.FloatTensor().to(device)
        preds_prob = torch.FloatTensor().to(device)
        preds = torch.FloatTensor().to(device)

        for batch_idx, (images, batch_labels) in enumerate(train_loader):
            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, batch_labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            output = torch.sigmoid(logits)
            preds_prob = torch.cat((preds_prob, output.data), 0)
            pred = (output.data > 0.5).type(torch.float32)
            preds = torch.cat((preds, pred.data), 0)
            labels = torch.cat((labels, batch_labels.data), 0)

            if batch_idx % 10 == 0:
                LOGGER.info(f"Batch: {batch_idx + 1}/{len(train_loader)}"
                            f" Loss: {running_loss / (batch_idx + 1):.4f}"
                            f" Time: {time.time() - start_time_epoch:2f}")
                if HPC_LOG and batch_idx % 300 == 0:
                    gpu_stats_dfs.append(log_gpu_utilization_csv(D_NAME, CLIENT_ID, ROUND, epoch, batch_idx))

        preds = preds.cpu().numpy().astype(np.int32)
        preds_prob = preds_prob.cpu().numpy()
        labels = labels.cpu().numpy().astype(np.int32)
        LOGGER.info("Training report:")
        LOGGER.info(classification_report(labels, preds, target_names=classes_names))

        aucs = {}
        for i, c in enumerate(classes_names):
            aucs[c] = roc_auc_score(labels.astype(np.float32)[:, i], preds_prob[:, i])

        avg_auc = np.mean(list(aucs.values()))

        test_loss = running_loss / len(train_loader)
        LOGGER.info(f" Loss: {test_loss:.4f}")
        LOGGER.info(f" Avg AUC: {avg_auc:.4f}")
        LOGGER.info(f" AUCs: {aucs}")
        if HPC_LOG:
            save_round_gpu_csv(gpu_stats_dfs, SERVER_ADDRESS, D_NAME, str(CLIENT_ID), ROUND)


def load_data(client_id, clients_number, d_name, bs):
    images_dir, train_subset, _, _ = get_data_paths(d_name)
    LOGGER.info(f"images_dir: {images_dir}")
    df = pd.read_csv(train_subset)
    dataset_len = len(df)
    selector = IIDSelector()
    ids = selector.get_ids(dataset_len, client_id, clients_number)

    decoder = RandomResizedCropRGBImageDecoder((224, 224))

    image_pipeline = [decoder, ToTensor(), ToDevice(device), ToTorchImage(),
                      Convert(target_dtype=torch.float32),
                      torchvision.transforms.Normalize(mean=[123.675, 116.28, 103.53],
                                                       std=[58.395, 57.12, 57.375])]

    if get_type_of_dataset(d_name) == 'multi-class':
        label_pipeline = [NDArrayDecoder(), ToTensor(), ToDevice(device)]
    else:
        label_pipeline = [IntDecoder(), ToTensor(), ToDevice(device), Squeeze()]

    pipelines = {
        'image': image_pipeline,
        'label': label_pipeline
    }
    train_subset_beton, _ = get_beton_data_paths(d_name)
    train_loader = Loader(train_subset_beton, batch_size=bs, num_workers=8, order=OrderOption.SEQUENTIAL,
                          pipelines=pipelines, indices=ids)

    return train_loader, get_class_names(d_name)


class ClassificationClient(fl.client.NumPyClient):
    def __init__(self, client_id, clients_number, m_name):
        # Load model
        # TODO now 14 classes is hardcoded, should be possible to config
        self.model = get_model(m_name, classes=14)
        self.client_id = client_id
        self.clients_number = clients_number
        self.train_loader = None
        self.classes_names = None

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        LOGGER.info("Loading parameters...")
        state_dict = get_state_dict(self.model, parameters)
        self.model.load_state_dict(state_dict, strict=True)
        LOGGER.info("Parameters loaded")

    def fit(self, parameters, config):
        global D_NAME, ROUND, HPC_LOG
        self.set_parameters(parameters)

        batch_size = int(config["batch_size"])
        epochs = int(config["local_epochs"])
        lr = float(config["learning_rate"])
        D_NAME = d_name = config["dataset_type"]
        ROUND = config["round_no"]
        HPC_LOG = config["hpc_log"]

        LOGGER.info(f"Learning rate: {lr}")

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.00001)
        self.train_loader, self.classes_names = load_data(self.client_id, self.clients_number, d_name, batch_size)

        if get_type_of_dataset(d_name) == 'multi-class':
            criterion = nn.BCEWithLogitsLoss()
            train_multi_label(self.model, self.train_loader, criterion, optimizer, self.classes_names, epochs=epochs)
        else:
            criterion = nn.CrossEntropyLoss()
            train_single_label(self.model, self.train_loader, criterion, optimizer, self.classes_names, epochs=epochs)

        return self.get_parameters(), len(self.train_loader), {}

    def evaluate(self, parameters, config):
        # TODO this is not reported at the moment
        pass


@click.command()
@click.option('--sa', default='', type=str, help='Server address')
@click.option('--c_id', default=0, type=int, help='Client id')
@click.option('--c', default=1, type=int, help='Clients number')
@click.option('--m', default='DenseNet121', type=str, help='Model used for training')
def run_client(sa, c_id, c, m):
    global CLIENT_ID
    global SERVER_ADDRESS
    # Start client
    LOGGER.info(f"Cpu count: {os.cpu_count()}")
    LOGGER.info("Connecting to:" + f"{sa}:8087")
    CLIENT_ID = c_id
    SERVER_ADDRESS = sa
    fl.client.start_numpy_client(f"{sa}:8087",
                                 client=ClassificationClient(c_id, c, m))


if __name__ == "__main__":
    run_client()
