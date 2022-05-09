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
from collections import defaultdict
import numpy as np

from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToDevice, ToTorchImage, NormalizeImage, Convert, ToTensor
from ffcv.transforms.common import Squeeze
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder, NDArrayDecoder

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
LEARNING_RATE = 0.0001
MIN_LEARNING_RATE = 0.000001
PATIENCE = 2
CURRENT_PATIENCE = 2
MODEL_NAME = 'DenseNet121'
DATASETS_TYPE = 'nih'
SERVER_ADDR = ''
HPC_LOG = False
CLIENT_JOB_IDS = []
HPC_METRICS_DF = None
GPU_METRICS = []
DOWNSAMPLE_TEST = True
DATA_SELECTION = 'iid'
TEST_DATASETS = ['nih']
STUDY_PREFIX = 'nih'
RESULTS_BUCKET = 'fl-msc-classification'

IMAGE_SIZE = 224
LIMIT = -1

TIME_START = time.time()

loss_dict = defaultdict(list)
acc_dict = defaultdict(list)
avg_auc_dict = defaultdict(list)
aucs_dict = defaultdict(list)
reports_dict = defaultdict(list)
times = []
learning_rates = []
downsample_test = []


def fit_config(rnd: int):
    config = {
        "batch_size": BATCH_SIZE,
        "local_epochs": LOCAL_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "round_no": ROUND,
        "hpc_log": HPC_LOG,
        "data_selection": DATA_SELECTION,
        "model_name": MODEL_NAME
    }
    return config


def results_dirname_generator():
    return f'd_{DATASETS_TYPE}_m_{MODEL_NAME}_r_{MAX_ROUNDS}-c_{CLIENTS}_bs_{BATCH_SIZE}_le_{LOCAL_EPOCHS}' \
           f'_mf_{MIN_FIT_CLIENTS}_ff_{FRACTION_FIT}_image_{IMAGE_SIZE}_data_selection_{DATA_SELECTION}_test_datasets_{"#".join(TEST_DATASETS)}'


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
        client_gpu_metrics_dir = os.path.join(gpu_metrics_dir, f'{DATASETS_TYPE}_{client_id}')
        os.mkdir(client_gpu_metrics_dir)
    server_gpu_metrics_dir = os.path.join(gpu_metrics_dir, f'{DATASETS_TYPE}_server')
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
    def __init__(self, le, c, mf, ff, bs, lr, m, test_datasets):
        self.le = le
        self.c = c
        self.mf = mf
        self.ff = ff
        self.bs = bs
        self.lr = lr
        self.test_datasets = test_datasets
        self.model = get_model(m, classes=get_dataset_classes_count(self.test_datasets[0]))

    def get_eval_fn(self, model):
        image_pipeline = [RandomResizedCropRGBImageDecoder((224, 224), scale=(1.0, 1.0), ratio=(1.0, 1.0)), ToTensor(),
                          ToDevice(DEVICE), ToTorchImage(),
                          Convert(target_dtype=torch.float32),
                          torchvision.transforms.Normalize(mean=[123.675, 116.28, 103.53],
                                                           std=[58.395, 57.12, 57.375])]

        if get_type_of_dataset(self.test_datasets[0]) == 'multi-class':
            label_pipeline = [NDArrayDecoder(), ToTensor(), ToDevice(DEVICE)]
            criterion = nn.BCEWithLogitsLoss()
        else:
            label_pipeline = [IntDecoder(), ToTensor(), ToDevice(DEVICE), Squeeze()]
            criterion = nn.CrossEntropyLoss()

        pipelines = {
            'image': image_pipeline,
            'label': label_pipeline
        }

        test_loaders_dict = dict()

        for t_dataset in self.test_datasets:
            _, test_subset = get_beton_data_paths(t_dataset)
            LOGGER.info(f"test_subset_file: {test_subset}")
            if DOWNSAMPLE_TEST:
                _, _, test_subset_list, _ = get_data_paths(t_dataset)
                df = pd.read_csv(test_subset_list)
                dataset_len = len(df)
                selector = IIDSelector()
                # Downsample dataset by factor of 10
                ids = selector.get_ids(dataset_len, 0, 10)

                test_loaders_dict[t_dataset] = Loader(test_subset, batch_size=BATCH_SIZE, num_workers=1,
                                                      order=OrderOption.SEQUENTIAL, pipelines=pipelines, indices=ids,
                                                      drop_last=False)
            else:
                test_loaders_dict[t_dataset] = Loader(test_subset, batch_size=BATCH_SIZE, num_workers=1,
                                                      order=OrderOption.SEQUENTIAL, pipelines=pipelines,
                                                      drop_last=False)

        # Assumes all datasets have the same classes
        classes_names = get_class_names(self.test_datasets[0])

        def evaluate(weights):
            global ROUND, LEARNING_RATE, PATIENCE, CURRENT_PATIENCE, RESULTS_BUCKET
            state_dict = get_state_dict(model, weights)
            model.load_state_dict(state_dict, strict=True)

            for t_dataset in TEST_DATASETS:
                if get_type_of_dataset(t_dataset) == 'multi-class':
                    test_avg_auc, test_loss, report_json, auc_json = test_multi_label(model, LOGGER,
                                                                                      test_loaders_dict[t_dataset],
                                                                                      criterion,
                                                                                      classes_names, SERVER_ADDR,
                                                                                      t_dataset,
                                                                                      'server', ROUND, HPC_LOG)
                    avg_auc_dict[t_dataset].append(test_avg_auc)
                    aucs_dict[t_dataset].append(auc_json)
                else:
                    test_acc, test_loss, report_json = test_single_label(model, LOGGER, test_loaders_dict[t_dataset],
                                                                         criterion,
                                                                         classes_names, SERVER_ADDR, t_dataset,
                                                                         'server',
                                                                         ROUND, HPC_LOG)
                    acc_dict[t_dataset].append(test_acc)

                loss_dict[t_dataset].append(test_loss)
                reports_dict[t_dataset].append(report_json)

            times.append(time.time() - TIME_START)
            learning_rates.append(LEARNING_RATE)
            downsample_test.append(DOWNSAMPLE_TEST)

            should_decrease_lr = True

            for t_dataset in self.test_datasets:
                should_decrease_lr = should_decrease_lr and (
                        len(loss_dict[t_dataset][:-1]) != 0 and loss_dict[t_dataset][-1] >= min(
                    loss_dict[t_dataset][:-1]))

            if should_decrease_lr:
                CURRENT_PATIENCE -= 1
                if CURRENT_PATIENCE == 0:
                    LEARNING_RATE = max(LEARNING_RATE / 10, MIN_LEARNING_RATE)
                    LOGGER.info(f"No improvement in loss, updating learning rate, now lr= {LEARNING_RATE}")
                    CURRENT_PATIENCE = PATIENCE
            else:
                CURRENT_PATIENCE = PATIENCE

            if get_type_of_dataset(self.test_datasets[0]) == 'multi-class':
                res_dict = {}
                for t_dataset in self.test_datasets:
                    res_dict = res_dict | {f'{t_dataset}#loss': loss_dict[t_dataset],
                                           f'{t_dataset}#avg_auc': avg_auc_dict[t_dataset],
                                           f'{t_dataset}#aucs': aucs_dict[t_dataset],
                                           f'{t_dataset}#report': reports_dict[t_dataset]}
            else:
                res_dict = {}
                for t_dataset in self.test_datasets:
                    res_dict = res_dict | {f'{t_dataset}#loss': loss_dict[t_dataset],
                                           f'{t_dataset}#acc': acc_dict[t_dataset],
                                           f'{t_dataset}#report': reports_dict[t_dataset]}

            df = pd.DataFrame.from_dict(
                {'round': [i for i in range(ROUND + 1)], 'time': times, 'lr': learning_rates,
                 'downsample_test': downsample_test} | res_dict)

            res_dir = results_dirname_generator()

            is_best_model = True
            for t_dataset in self.test_datasets:
                is_best_model = is_best_model and (
                        len(loss_dict[t_dataset][:-1]) != 0 and loss_dict[t_dataset][-1] < min(
                    loss_dict[t_dataset][:-1]))

            if is_best_model:
                model_dir = os.path.join(res_dir, 'best_model')
                if os.path.exists(model_dir):
                    shutil.rmtree(model_dir)
                os.mkdir(model_dir)
                LOGGER.info(f"Saving model as all loses are the lowest")
                torch.save(model.state_dict(),
                           f'{model_dir}/{MODEL_NAME}_{ROUND}')

            df['data_selection'] = DATA_SELECTION
            df.to_csv(os.path.join(res_dir, 'result.csv'))

            if HPC_LOG:
                log_hpc_usage(os.environ["SLURM_JOB_ID"])
                update_gpu_stats()

            ROUND += 1

            if RESULTS_BUCKET:
                copy_status = subprocess.run(
                    ['gsutil', '-m', 'cp', '-r', f'{res_dir}/', f'gs://{RESULTS_BUCKET}/{STUDY_PREFIX}/'],
                    stdout=subprocess.PIPE).stdout.decode('utf-8')
                LOGGER.info(copy_status)

            if get_type_of_dataset(self.test_datasets[0]) == 'multi-class':
                return np.mean([loss_dict[d][-1] for d in self.test_datasets]), {
                    "test_avg_auc": np.mean([avg_auc_dict[d][-1] for d in self.test_datasets])}
            else:
                return np.mean([loss_dict[d][-1] for d in self.test_datasets]), {
                    "test_acc": np.mean([acc_dict[d][-1] for d in self.test_datasets])}

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
@click.option('--d', default='nih', type=str, help='Datasets used for training (nih), comma separated')
@click.option('--hpc-log', is_flag=True, help='Whether to log HPC usage metrics')
@click.option('--downsample-test', is_flag=True, help='Whether to downsample test set (to speed up FL process)')
@click.option('--data-selection', default='iid', type=str, help='Kind of data selection strategy for clients')
@click.option('--test_datasets', default='nih', type=str,
              help='List of datasets used for evaluation of global model, comma separated')
@click.option('--results_bucket', default='fl-msc-classification', type=str, help='GCP bucket for storing results')
@click.option('--study_prefix', default='nih', type=str, help='General name of the experiment')
def run_server(le, c, r, mf, ff, bs, lr, m, d, hpc_log, downsample_test, data_selection, test_datasets, results_bucket,
               study_prefix):
    global LOCAL_EPOCHS, CLIENTS, MAX_ROUNDS, MIN_FIT_CLIENTS, FRACTION_FIT, BATCH_SIZE, LEARNING_RATE, MODEL_NAME, \
        DATASETS_TYPE, HPC_LOG, SERVER_ADDR, DOWNSAMPLE_TEST, DATA_SELECTION, TEST_DATASETS, RESULTS_BUCKET, STUDY_PREFIX

    LOCAL_EPOCHS = le
    CLIENTS = c
    MAX_ROUNDS = r
    MIN_FIT_CLIENTS = mf
    FRACTION_FIT = ff
    BATCH_SIZE = bs
    LEARNING_RATE = lr
    MODEL_NAME = m
    DATASETS_TYPE = '#'.join(sorted(d.split(',')))
    HPC_LOG = hpc_log
    DOWNSAMPLE_TEST = downsample_test
    DATA_SELECTION = data_selection
    SERVER_ADDR = server_addr = socket.gethostname()
    TEST_DATASETS = sorted(test_datasets.split(','))
    RESULTS_BUCKET = results_bucket
    STUDY_PREFIX = study_prefix

    factory = StrategyFactory(le, c, mf, ff, bs, lr, m, TEST_DATASETS)
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
