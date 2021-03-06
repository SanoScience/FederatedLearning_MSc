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
from ffcv.transforms import ToDevice, ToTorchImage, Cutout, NormalizeImage, Convert, ToTensor, RandomHorizontalFlip, \
    RandomTranslate
from ffcv.transforms.common import Squeeze
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder, NDArrayDecoder

from data_selector import IIDSelector, IncreasingSelector, NonIIDSelector
from fl_cc_cxri_p_dataset import CCCXRIPDataset

from utils import get_state_dict, accuracy, get_model, get_data_paths, get_beton_data_paths, \
    get_type_of_dataset, get_class_names, log_gpu_utilization_csv, make_round_gpu_metrics_dir, save_round_gpu_csv, \
    CC_CXRI_P_CLASSES, get_dataset_classes_count, get_convert_label_fun, get_train_transform_rsna

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
HPC_LOG_FREQUENCY = 100
REDUCED_CLASSES = False

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
            images = images.to(device=device, dtype=torch.float32)
            batch_labels = batch_labels.to(device=device)

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
                if HPC_LOG and batch_idx % HPC_LOG_FREQUENCY == 0:
                    gpu_stats_dfs.append(log_gpu_utilization_csv(D_NAME, CLIENT_ID, ROUND, epoch, batch_idx))
        preds = preds.cpu().numpy().astype(np.int32)
        labels = labels.cpu().numpy().astype(np.int32)
        # LOGGER.info("Training report:")
        # LOGGER.info(classification_report(labels, preds, target_names=classes_names))

        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(labels, preds)

        LOGGER.info(f" Training Loss: {train_loss:.4f}"
                    f" Training Acc: {train_acc:.4f}")
        if HPC_LOG:
            save_round_gpu_csv(gpu_stats_dfs, SERVER_ADDRESS, D_NAME, str(CLIENT_ID), ROUND)


def train_multi_label(model, train_loader, criterion, optimizer, classes_names, epochs, convert_label_fun):
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

            # U-zeros approach
            batch_labels[batch_labels != 1.0] = 0.0
            batch_labels = convert_label_fun(batch_labels)

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
                if HPC_LOG and batch_idx % HPC_LOG_FREQUENCY == 0:
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


def load_data(client_id, clients_number, d_name, bs, data_selection='iid'):
    _, train_subset, _, _ = get_data_paths(d_name)
    df = pd.read_csv(train_subset)

    dataset_len = len(df)
    if data_selection == 'iid':
        selector = IIDSelector()
        ids = selector.get_ids(dataset_len, client_id, clients_number)
    elif data_selection == 'increasing':
        selector = IncreasingSelector()
        ids = selector.get_ids(dataset_len, client_id, clients_number)
    elif data_selection == 'noniid':
        # Supports only CC_CXRI_P at the moment
        selector = NonIIDSelector()
        ids = selector.get_ids(df, client_id, CC_CXRI_P_CLASSES)

    decoder = RandomResizedCropRGBImageDecoder((224, 224), scale=(0.5, 1.0), ratio=(0.75, 4 / 3))

    IMAGENET_MEAN = [123.675, 116.28, 103.53]
    IMAGENET_STD = [58.395, 57.12, 57.375]

    image_pipeline = [decoder,
                      RandomHorizontalFlip(flip_prob=0.5),
                      # RandomTranslate(padding=10),
                      # Cutout(32, tuple(map(int, IMAGENET_MEAN))),
                      ToTensor(), ToDevice(device), ToTorchImage(),
                      Convert(target_dtype=torch.float32),
                      torchvision.transforms.Normalize(mean=IMAGENET_MEAN,
                                                       std=IMAGENET_STD)]

    if get_type_of_dataset(d_name) == 'multi-class':
        label_pipeline = [NDArrayDecoder(), ToTensor(), ToDevice(device)]
    else:
        label_pipeline = [IntDecoder(), ToTensor(), ToDevice(device), Squeeze()]

    pipelines = {
        'image': image_pipeline,
        'label': label_pipeline
    }
    train_subset_beton, _ = get_beton_data_paths(d_name)

    trans = get_train_transform_rsna(224)
    train_dataset = CCCXRIPDataset(client_id, clients_number, '/home/filip_slazyk/fl-datasets/CC-CXRI-P/train.csv',
                                  "/home/filip_slazyk/fl-datasets/CC-CXRI-P/cc-cxri-p-resized-224",
                                  trans)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=4)

    # train_loader = Loader(train_subset_beton, batch_size=bs, num_workers=8, order=OrderOption.SEQUENTIAL,
    #                       pipelines=pipelines, indices=ids, drop_last=False)

    return train_loader, get_class_names(d_name, REDUCED_CLASSES)


class ClassificationClient(fl.client.NumPyClient):
    def __init__(self, client_id, clients_number, dataset):
        self.client_id = client_id
        self.clients_number = clients_number
        self.train_loader = None
        self.classes_names = None
        self.model = None
        self.dataset_name = dataset

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        LOGGER.info("Loading parameters...")
        state_dict = get_state_dict(self.model, parameters)
        self.model.load_state_dict(state_dict, strict=True)
        LOGGER.info("Parameters loaded")

    def fit(self, parameters, config):
        global D_NAME, ROUND, HPC_LOG

        batch_size = int(config["batch_size"])
        epochs = int(config["local_epochs"])
        lr = float(config["learning_rate"])
        D_NAME = self.dataset_name
        ROUND = config["round_no"]
        HPC_LOG = config["hpc_log"]
        data_selection = config["data_selection"]
        model_name = config["model_name"]
        reduced_classes = config["reduced_classes"]

        if self.model is None:
            self.model = get_model(model_name, classes=get_dataset_classes_count(self.dataset_name, REDUCED_CLASSES))

        self.set_parameters(parameters)

        LOGGER.info(f"Learning rate: {lr}")

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.00001)
        # optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=0)
        self.train_loader, self.classes_names = load_data(self.client_id, self.clients_number, self.dataset_name,
                                                          batch_size,
                                                          data_selection=data_selection)

        if get_type_of_dataset(self.dataset_name) == 'multi-class':
            criterion = nn.BCEWithLogitsLoss()
            convert_label_fun = lambda x: x
            if reduced_classes:
                convert_label_fun = get_convert_label_fun(self.dataset_name)
            train_multi_label(self.model, self.train_loader, criterion, optimizer, self.classes_names, epochs=epochs,
                              convert_label_fun=convert_label_fun)
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
@click.option('--d', default='nih', type=str, help='Dataset on client')
@click.option('--reduced_classes', is_flag=True, help='Use subset of 7 common classes')
def run_client(sa, c_id, c, d, reduced_classes):
    global CLIENT_ID, SERVER_ADDRESS, REDUCED_CLASSES
    # Start client
    LOGGER.info(f"Cpu count: {os.cpu_count()}")
    LOGGER.info("Connecting to:" + f"{sa}:8087")
    CLIENT_ID = c_id
    SERVER_ADDRESS = sa
    REDUCED_CLASSES = reduced_classes
    fl.client.start_numpy_client(f"{sa}:8087",
                                 client=ClassificationClient(c_id, c, d))


if __name__ == "__main__":
    run_client()
