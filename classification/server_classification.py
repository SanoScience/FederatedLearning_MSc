import flwr as fl
import socket
import logging
import torch
import torch.nn as nn
import pandas as pd
import click
import time
import os
import shutil
from io import StringIO
import subprocess

from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToDevice, ToTorchImage, NormalizeImage, Convert, ToTensor
from ffcv.transforms.common import Squeeze
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder, NDArrayDecoder

import torchvision
import glob

from data_selector import IIDSelector
from utils import get_state_dict, test_single_label, get_beton_data_paths, get_model, get_class_names, \
    get_type_of_dataset, get_dataset_classes_count, test_multi_label, get_data_paths

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LOGGER = logging.getLogger(__name__)
hdlr = logging.StreamHandler()
LOGGER.addHandler(hdlr)
LOGGER.setLevel(logging.INFO)

ROUND = 0

LOCAL_EPOCHS = 1
CLIENTS = 3
MAX_ROUNDS = 20
MIN_FIT_CLIENTS = 3
FRACTION_FIT = 0.1
BATCH_SIZE = 8
LEARNING_RATE = 0.01
MODEL_NAME = 'DenseNet121'
DATASET_TYPE = 'nih'
SERVER_ADDR = ''
HPC_LOG = False
CLIENT_JOB_IDS = []
HPC_METRICS_DF = None
GPU_METRICS = []
DOWNSAMPLE_TEST = True

IMAGE_SIZE = 224
LIMIT = -1

TIME_START = time.time()

loss = []
acc = []
avg_auc = []
aucs = []
reports = []
times = []
learning_rates = []
downsample_test = []


def fit_config(rnd: int):
    config = {
        "batch_size": BATCH_SIZE,
        "local_epochs": LOCAL_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "dataset_type": DATASET_TYPE,
        "round_no": ROUND,
        "hpc_log": HPC_LOG
    }
    return config


def results_dirname_generator():
    return f'd_{DATASET_TYPE}_m_{MODEL_NAME}_r_{MAX_ROUNDS}-c_{CLIENTS}_bs_{BATCH_SIZE}_le_{LOCAL_EPOCHS}' \
           f'_mf_{MIN_FIT_CLIENTS}_ff_{FRACTION_FIT}_lr_{LEARNING_RATE}_image_{IMAGE_SIZE}_IID'


def get_slurm_stats(job_id, job_type, node_id):
    metrics = subprocess.run(['sstat', job_id, '--parsable2', '--noconvert'], stdout=subprocess.PIPE)
    metrics_string = metrics.stdout.decode('utf-8')
    df = pd.read_csv(StringIO(metrics_string), delimiter='|')
    df['job_type'] = job_type
    df['round'] = ROUND
    df['node_id'] = node_id
    return df


def update_gpu_stats():
    global GPU_METRICS
    server_addr = socket.gethostname()
    gpu_metrics_dir = f'gpu_metrics_cache_{server_addr}'
    round_path = os.path.join(gpu_metrics_dir, '*', str(ROUND), '*.csv')
    csv_files = glob.glob(round_path)
    round_df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
    GPU_METRICS.append(round_df)
    round_merged_df = pd.concat(GPU_METRICS, ignore_index=True)

    res_dir = results_dirname_generator()
    gpu_final_metrics_path = os.path.join(res_dir, 'gpu_metrics', f'gpu_metrics_{ROUND}.csv')
    round_merged_df.to_csv(gpu_final_metrics_path)


def make_gpu_usage_dirs():
    server_addr = socket.gethostname()
    gpu_metrics_dir = f'gpu_metrics_cache_{server_addr}'
    if os.path.exists(gpu_metrics_dir):
        shutil.rmtree(gpu_metrics_dir)
    os.mkdir(gpu_metrics_dir)
    res_dir = results_dirname_generator()
    gpu_final_metrics_dir = os.path.join(res_dir, 'gpu_metrics')
    if os.path.exists(gpu_final_metrics_dir):
        shutil.rmtree(gpu_final_metrics_dir)
    os.mkdir(gpu_final_metrics_dir)
    for client_id in range(CLIENTS):
        client_gpu_metrics_dir = os.path.join(gpu_metrics_dir, f'{DATASET_TYPE}_{client_id}')
        os.mkdir(client_gpu_metrics_dir)
    server_gpu_metrics_dir = os.path.join(gpu_metrics_dir, f'{DATASET_TYPE}_server')
    os.mkdir(server_gpu_metrics_dir)


def log_hpc_usage(server_job_id):
    global CLIENT_JOB_IDS
    global HPC_METRICS_DF

    res_dir = results_dirname_generator()
    hpc_usage_dir = os.path.join(res_dir, 'hpc_metrics')
    if ROUND == 0:
        if os.path.exists(hpc_usage_dir):
            shutil.rmtree(hpc_usage_dir)
        os.mkdir(hpc_usage_dir)
    else:
        if not CLIENT_JOB_IDS:
            client_ids_file = f'{server_job_id}_client_ids.txt'
            with open(client_ids_file) as f:
                CLIENT_JOB_IDS = [line.strip() for line in f]

        server_metrics_df = get_slurm_stats(server_job_id, 'server', 0)

        if HPC_METRICS_DF is None:
            HPC_METRICS_DF = server_metrics_df
        else:
            HPC_METRICS_DF = pd.concat([HPC_METRICS_DF, server_metrics_df], ignore_index=True)

        for i, client_job_id in enumerate(CLIENT_JOB_IDS):
            client_metrics_df = get_slurm_stats(client_job_id, 'client', i)
            HPC_METRICS_DF = pd.concat([HPC_METRICS_DF, client_metrics_df], ignore_index=True)

        metrics_file = os.path.join(hpc_usage_dir, f'hpc_metrics_{ROUND}.csv')
        HPC_METRICS_DF.to_csv(metrics_file)


class StrategyFactory:
    def __init__(self, le, c, mf, ff, bs, lr, m, d):
        self.le = le
        self.c = c
        self.mf = mf
        self.ff = ff
        self.bs = bs
        self.lr = lr
        self.d = d
        self.model = get_model(m, classes=get_dataset_classes_count(self.d))

    def get_eval_fn(self, model):
        _, test_subset = get_beton_data_paths(self.d)
        LOGGER.info(f"images_dir: {test_subset}")

        image_pipeline = [SimpleRGBImageDecoder(), ToTensor(), ToDevice(DEVICE), ToTorchImage(),
                          Convert(target_dtype=torch.float32),
                          torchvision.transforms.Normalize(mean=[123.675, 116.28, 103.53],
                                                           std=[58.395, 57.12, 57.375])]

        if get_type_of_dataset(self.d) == 'multi-class':
            label_pipeline = [NDArrayDecoder(), ToTensor(), ToDevice(DEVICE)]
            criterion = nn.BCEWithLogitsLoss()
        else:
            label_pipeline = [IntDecoder(), ToTensor(), ToDevice(DEVICE), Squeeze()]
            criterion = nn.CrossEntropyLoss()

        pipelines = {
            'image': image_pipeline,
            'label': label_pipeline
        }

        if DOWNSAMPLE_TEST:
            images_dir, _, test_subset, _ = get_data_paths(DATASET_TYPE)
            df = pd.read_csv(test_subset)
            dataset_len = len(df)
            selector = IIDSelector()
            ids = selector.get_ids(dataset_len, 0, 10)

            test_loader = Loader(test_subset, batch_size=BATCH_SIZE, num_workers=12, order=OrderOption.SEQUENTIAL,
                                 pipelines=pipelines, ids=ids)
        else:
            test_loader = Loader(test_subset, batch_size=BATCH_SIZE, num_workers=12, order=OrderOption.SEQUENTIAL,
                                 pipelines=pipelines)

        classes_names = get_class_names(self.d)

        def evaluate(weights):
            global ROUND, LEARNING_RATE
            state_dict = get_state_dict(model, weights)
            model.load_state_dict(state_dict, strict=True)

            if get_type_of_dataset(self.d) == 'multi-class':
                test_avg_auc, test_loss, report_json, auc_json = test_multi_label(model, LOGGER, test_loader, criterion,
                                                                                  classes_names, SERVER_ADDR, self.d,
                                                                                  'server', ROUND, HPC_LOG)
                avg_auc.append(test_avg_auc)
                aucs.append(auc_json)
            else:
                test_acc, test_loss, report_json = test_single_label(model, LOGGER, test_loader, criterion,
                                                                     classes_names, SERVER_ADDR, self.d, 'server',
                                                                     ROUND, HPC_LOG)
                acc.append(test_acc)

            loss.append(test_loss)
            reports.append(report_json)
            times.append(time.time() - TIME_START)
            learning_rates.append(LEARNING_RATE)
            downsample_test.append(DOWNSAMPLE_TEST)

            if len(loss[:-1]) != 0 and test_loss >= min(loss[:-1]):
                LEARNING_RATE /= 10
                LOGGER.info(f"No improvement in loss, updating learning rate, now lr= {LEARNING_RATE}")

            if get_type_of_dataset(self.d) == 'multi-class':
                df = pd.DataFrame.from_dict(
                    {'round': [i for i in range(ROUND + 1)], 'loss': loss, 'avg_auc': avg_auc, 'aucs': aucs,
                     'report': reports, 'time': times, 'lr': learning_rates, 'downsample_test': downsample_test})
            else:
                df = pd.DataFrame.from_dict(
                    {'round': [i for i in range(ROUND + 1)], 'loss': loss, 'acc': acc, 'report': reports,
                     'time': times, 'lr': learning_rates, 'downsample_test': downsample_test})

            res_dir = results_dirname_generator()
            if len(loss[:-1]) != 0 and test_loss < min(loss[:-1]):
                model_dir = os.path.join(res_dir, 'best_model')
                if os.path.exists(model_dir):
                    shutil.rmtree(model_dir)
                os.mkdir(model_dir)
                LOGGER.info(f"Saving model as loss is the lowest: {test_loss}")
                torch.save(model.state_dict(),
                           f'{model_dir}/{MODEL_NAME}_{ROUND}_loss_{round(test_loss, 3)}')

            df.to_csv(os.path.join(res_dir, 'result.csv'))

            if HPC_LOG:
                log_hpc_usage(os.environ["SLURM_JOB_ID"])
                update_gpu_stats()

            ROUND += 1

            if get_type_of_dataset(self.d) == 'multi-class':
                return test_loss, {"test_avg_auc": test_avg_auc}
            else:
                return test_loss, {"test_acc": test_acc}

        return evaluate

    def get_strategy(self):
        return fl.server.strategy.FedAvg(
            fraction_fit=FRACTION_FIT,
            fraction_eval=0.5,
            min_fit_clients=MIN_FIT_CLIENTS,
            min_eval_clients=1,
            min_available_clients=CLIENTS,
            eval_fn=self.get_eval_fn(self.model),
            on_fit_config_fn=fit_config,
            initial_parameters=fl.common.weights_to_parameters(
                [val.cpu().numpy() for _, val in self.model.state_dict().items()])
        )


@click.command()
@click.option('--le', default=LOCAL_EPOCHS, type=int, help='Local epochs performed by clients')
@click.option('--c', default=CLIENTS, type=int, help='Clients number')
@click.option('--r', default=MAX_ROUNDS, type=int, help='Rounds of training')
@click.option('--mf', default=MIN_FIT_CLIENTS, type=int, help='Min fit clients')
@click.option('--ff', default=FRACTION_FIT, type=float, help='Fraction fit')
@click.option('--bs', default=BATCH_SIZE, type=int, help='Batch size')
@click.option('--lr', default=LEARNING_RATE, type=float, help='Learning rate')
@click.option('--m', default='DenseNet121', type=str, help='Model used for training')
@click.option('--d', default='nih', type=str, help='Dataset used for training (nih)')
@click.option('--hpc-log', is_flag=True, help='Whether to log HPC usage metrics')
@click.option('--downsample-test', is_flag=True, help='Whether to downsample test set (to speed up FL process)')
def run_server(le, c, r, mf, ff, bs, lr, m, d, hpc_log, downsample_test):
    global LOCAL_EPOCHS, CLIENTS, MAX_ROUNDS, MIN_FIT_CLIENTS, FRACTION_FIT, BATCH_SIZE, LEARNING_RATE, MODEL_NAME, \
        DATASET_TYPE, HPC_LOG, SERVER_ADDR, DOWNSAMPLE_TEST

    LOCAL_EPOCHS = le
    CLIENTS = c
    MAX_ROUNDS = r
    MIN_FIT_CLIENTS = mf
    FRACTION_FIT = ff
    BATCH_SIZE = bs
    LEARNING_RATE = lr
    MODEL_NAME = m
    DATASET_TYPE = d
    HPC_LOG = hpc_log
    DOWNSAMPLE_TEST = downsample_test
    SERVER_ADDR = server_addr = socket.gethostname()

    factory = StrategyFactory(le, c, mf, ff, bs, lr, m, d)
    strategy = factory.get_strategy()

    res_dir = results_dirname_generator()
    if os.path.exists(res_dir):
        shutil.rmtree(res_dir)
    os.mkdir(res_dir)

    if HPC_LOG:
        make_gpu_usage_dirs()

    # Start server
    LOGGER.info(f"Starting server on {server_addr}")
    fl.server.start_server(
        server_address=f"{server_addr}:8087",
        config={"num_rounds": MAX_ROUNDS},
        strategy=strategy,
    )


if __name__ == "__main__":
    run_server()
