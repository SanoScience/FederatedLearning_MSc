import os
import torch
from collections import OrderedDict
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

import torch.nn.functional as F

import torchvision
import json
import numpy as np
import subprocess
import pandas as pd
from io import StringIO

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

REPO_DATASETS_PATH_BASE = os.path.expandvars("$SCRATCH/FederatedLearning_MSc/classification/datasets")
RSNA_DATASET_PATH_BASE = os.path.expandvars("$SCRATCH/fl_msc/classification/RSNA/")
NIH_DATASET_PATH_BASE = os.path.join(REPO_DATASETS_PATH_BASE, "NIH/")
CC_CXRI_P_PATH_BASE = os.path.join(REPO_DATASETS_PATH_BASE, "CC-CXRI-P/")
CHESTDX_DATASET_PATH_BASE = os.path.expandvars(
    "$PLG_GROUPS_STORAGE/plggsano/fl_msc_classification/classification/China_X_ray")
CHEXPERT_DATASET_PATH_BASE = os.path.expandvars("$PLG_GROUPS_STORAGE/plggsano/Chexpert/Chexpert_dataset")
MIMIC_DATASET_PATH_BASE = os.path.expandvars(
    "$SCRATCH/fl_msc/classification/MIMIC/mimic-cxr-jpg-2.0.0.physionet.org/files")

NIH_CHESTDX_CLASSES = ["Consolidation", "Fibrosis", "Nodule", "Hernia", "Atelectasis", "Pneumothorax", "Edema",
                       "Pneumonia", "Emphysema", "Effusion", "Infiltration", "Pleural_Thickening", "Mass",
                       "Cardiomegaly"]

CHEXPERT_MIMIC_CLASSES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
                          "Fracture", "Lung Lesion", "Lung Opacity", "Pleural Effusion",
                          "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"]

CC_CXRI_P_CLASSES = ["Normal", "Viral", "COVID", "Other"]


def accuracy(y_pred, y_true):
    y_pred = F.softmax(y_pred, dim=1)
    top_p, top_class = y_pred.topk(1, dim=1)
    equals = top_class == y_true.view(*top_class.shape)
    return torch.mean(equals.type(torch.FloatTensor))


def get_state_dict(model, parameters):
    params_dict = []
    for i, k in enumerate(list(model.state_dict().keys())):
        p = parameters[i]
        if 'num_batches_tracked' in k:
            p = p.reshape(p.size)
        params_dict.append((k, p))
    return OrderedDict({k: torch.Tensor(v) for k, v in params_dict})


def get_train_transform_rsna(img_size):
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(img_size, img_size)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomAffine(
            degrees=[-5, 5],
            translate=[0.05, 0.05],
            scale=[0.95, 1.05],
            shear=[-5, 5],
        ),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])


def get_test_transform_rsna(img_size):
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(img_size, img_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])


def get_model(m, classes=3):
    if m == 'ResNet50':
        model = torchvision.models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(in_features=num_ftrs, out_features=classes)
        model = model.to(DEVICE)
        return model
    if m == 'DenseNet121':
        model = torchvision.models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features=num_ftrs, out_features=classes)
        model = model.to(DEVICE)
        return model


def get_data_paths(dataset):
    if 'rsna' in dataset:
        RSNA_DATASET_PATH_BASE = os.path.expandvars("$SCRATCH/fl_msc/classification/RSNA/")
        train_subset = os.path.join(RSNA_DATASET_PATH_BASE, "train_labels_stage_1.csv")
        test_subset = os.path.join(RSNA_DATASET_PATH_BASE, "test_labels_stage_1.csv")
        if dataset == 'rsna-full':
            images_dir = os.path.join(RSNA_DATASET_PATH_BASE, "stage_2_train_images_jpg/")
        else:
            images_dir = os.path.join(RSNA_DATASET_PATH_BASE, "masked_stage_2_train_images_09_01_1024/")
        return images_dir, train_subset, test_subset, None
    elif dataset == 'nih':
        images_dir = os.path.join(NIH_DATASET_PATH_BASE, 'images')
        train_subset = os.path.join(REPO_DATASETS_PATH_BASE, 'NIH/train_val_list.txt')
        test_subset = os.path.join(REPO_DATASETS_PATH_BASE, 'NIH/test_list.txt')
        labels_file = os.path.join(REPO_DATASETS_PATH_BASE, 'NIH/Data_Entry_2017_v2020.csv')
        return images_dir, train_subset, test_subset, labels_file
    elif dataset == 'chestdx':
        images_dir = CHESTDX_DATASET_PATH_BASE
        train_subset = os.path.join(REPO_DATASETS_PATH_BASE, 'ChestDx/train.csv')
        test_subset = os.path.join(REPO_DATASETS_PATH_BASE, 'ChestDx/test.csv')
        return images_dir, train_subset, test_subset, None
    elif dataset == 'chestdx-pe':
        images_dir = CHESTDX_DATASET_PATH_BASE
        test_subset = os.path.join(REPO_DATASETS_PATH_BASE, 'ChestDx-PE/full.csv')
        return images_dir, None, test_subset, None
    elif dataset == 'cc-cxri-p':
        images_dir = CHESTDX_DATASET_PATH_BASE
        train_subset = os.path.join(REPO_DATASETS_PATH_BASE, 'CC-CXRI-P/train.csv')
        test_subset = os.path.join(REPO_DATASETS_PATH_BASE, 'CC-CXRI-P/test.csv')
        return images_dir, train_subset, test_subset, None
    elif dataset == 'chexpert':
        images_dir = CHEXPERT_DATASET_PATH_BASE
        train_subset = os.path.join(REPO_DATASETS_PATH_BASE, 'CheXpert/frontal_train.csv')
        test_subset = os.path.join(REPO_DATASETS_PATH_BASE, 'CheXpert/frontal_valid.csv')
        return images_dir, train_subset, test_subset, None
    elif dataset == 'mimic':
        images_dir = MIMIC_DATASET_PATH_BASE
        train_subset = os.path.join(REPO_DATASETS_PATH_BASE, 'MIMIC/frontal_train.csv')
        test_subset = os.path.join(REPO_DATASETS_PATH_BASE, 'MIMIC/frontal_test.csv')
        return images_dir, train_subset, test_subset, None


def get_type_of_dataset(dataset):
    if dataset in ['rsna', 'cc-cxri-p']:
        return 'single-class'
    if dataset in ['nih', 'chestdx', 'chestdx-pe', 'chexpert', 'mimic']:
        return 'multi-class'


def get_dataset_classes_count(dataset):
    classes_counts = {
        'rsna': 3,
        'cc-cxri-p': 4,
        'nih': 14,
        'chestdx': 14,
        'chestdx-pe': 14,
        'chexpert': 13,
        'mimic': 13
    }
    return classes_counts[dataset]


def get_beton_data_paths(dataset):
    if dataset == 'rsna':
        train_subset = os.path.join(RSNA_DATASET_PATH_BASE, 'train-jpg90.beton')
        test_subset = os.path.join(RSNA_DATASET_PATH_BASE, 'test-jpg90.beton')
        return train_subset, test_subset
    if dataset == 'nih':
        train_subset = os.path.join(NIH_DATASET_PATH_BASE, 'nih-train-256-jpg90.beton')
        test_subset = os.path.join(NIH_DATASET_PATH_BASE, 'nih-test-256-jpg90.beton')
        return train_subset, test_subset
    if dataset == 'cc-cxri-p':
        train_subset = os.path.join(CC_CXRI_P_PATH_BASE, 'cc-cxri-p-train-256-jpg90.beton')
        test_subset = os.path.join(CC_CXRI_P_PATH_BASE, 'cc-cxri-p-test-256-jpg90.beton')
        return train_subset, test_subset


def get_class_names(dataset):
    dataset_to_names = {
        'rsna': ["Normal", "No Lung Opacity / Not Normal", "Lung Opacity"],
        'cc-cxri-p': ["Normal", "Viral", "COVID", "Other"],
        'nih': NIH_CHESTDX_CLASSES,
        'chestdx': NIH_CHESTDX_CLASSES,
        'chestdx-pe': NIH_CHESTDX_CLASSES,
        'chexpert': CHEXPERT_MIMIC_CLASSES,
        'mimic': CHEXPERT_MIMIC_CLASSES
    }
    return dataset_to_names[dataset]


def make_round_gpu_metrics_dir(server_addr, dataset_name, node_id, round_no):
    gpu_metrics_dir = f'gpu_metrics_cache_{server_addr}'
    node_gpu_metrics_dir = os.path.join(gpu_metrics_dir, f'{dataset_name}_{node_id}')
    os.mkdir(os.path.join(node_gpu_metrics_dir, str(round_no)))


def log_gpu_utilization_csv(dataset_name, node_id, round_no, epoch_no, batch_no):
    metrics_fields = ['timestamp', 'name', 'pstate', 'memory.total', 'memory.used', 'memory.free', 'utilization.gpu',
                      'utilization.memory', 'encoder.stats.sessionCount', 'encoder.stats.averageFps',
                      'encoder.stats.averageLatency', 'temperature.gpu', 'temperature.memory', 'power.draw',
                      'power.limit', 'clocks.current.graphics', 'clocks.current.sm', 'clocks.current.memory']
    format_options = ['csv', 'nounits']

    metrics_string = subprocess.run(
        ['nvidia-smi', f'--format={",".join(format_options)}', f'--query-gpu={",".join(metrics_fields)}'],
        stdout=subprocess.PIPE).stdout.decode('utf-8')
    df = pd.read_csv(StringIO(metrics_string), delimiter=',')
    df['dataset'] = dataset_name
    df['node_id'] = node_id
    df['round_no'] = round_no
    df['epoch_no'] = epoch_no
    df['batch_no'] = batch_no
    return df


def save_round_gpu_csv(gpu_stats_dfs, server_address, dataset_name, node_id, round_no):
    gpu_metrics_dir = f'gpu_metrics_cache_{server_address}'
    node_gpu_metrics_dir = os.path.join(gpu_metrics_dir, f'{dataset_name}_{node_id}')
    gpu_metrics_file = os.path.join(node_gpu_metrics_dir, str(round_no),
                                    f'gpu_metrics_{dataset_name}_{node_id}_round_{round_no}.csv')
    gpu_metrics_df = pd.concat(gpu_stats_dfs, ignore_index=True)
    gpu_metrics_df.to_csv(gpu_metrics_file, index=False)


def test_single_label(model, logger, test_loader, criterion, classes_names, server_address, d_name, client_id,
                      round_no, hpc_log):
    gpu_stats_dfs = []
    if hpc_log:
        make_round_gpu_metrics_dir(server_address, d_name, client_id, round_no)
    test_running_loss = 0.0
    test_running_accuracy = 0.0
    model.eval()
    logger.info("Testing: ")

    test_labels = torch.IntTensor().to(DEVICE)
    test_preds = torch.IntTensor().to(DEVICE)

    with torch.no_grad():
        for batch_idx, (image, batch_label) in enumerate(test_loader):
            logits = model(image)
            loss = criterion(logits, batch_label)

            test_running_loss += loss.item()
            test_running_accuracy += accuracy(logits, batch_label)

            y_pred = F.softmax(logits, dim=1)
            _, top_class = y_pred.topk(1, dim=1)

            test_preds = torch.cat((test_preds, top_class.data), 0)
            test_labels = torch.cat((test_labels, batch_label.data), 0)

            if batch_idx % 50 == 0:
                logger.info(f"batch_idx: {batch_idx}\n"
                            f"running_loss: {test_running_loss / (batch_idx + 1):.4f}\n"
                            f"running_acc: {test_running_accuracy / (batch_idx + 1):.4f}\n\n")
                if hpc_log and batch_idx % 300 == 0:
                    gpu_stats_dfs.append(log_gpu_utilization_csv(d_name, client_id, round_no, 0, batch_idx))

    test_preds = test_preds.cpu().numpy().astype(np.int32)
    test_labels = test_labels.cpu().numpy().astype(np.int32)
    logger.info("Test report:")
    report = classification_report(test_labels, test_preds, target_names=classes_names)
    logger.info(report)
    report_json = json.dumps(
        classification_report(test_labels, test_preds, target_names=classes_names, output_dict=True))

    test_loss = test_running_loss / len(test_loader)
    test_acc = accuracy_score(test_labels, test_preds)
    logger.info(f" Test Loss: {test_loss:.4f}"
                f" Test Acc: {test_acc:.4f}")
    if hpc_log:
        save_round_gpu_csv(gpu_stats_dfs, server_address, d_name, str(client_id), round_no)

    return test_acc, test_loss, report_json


def test_multi_label(model, logger, test_loader, criterion, classes_names, server_address, d_name, client_id,
                     round_no, hpc_log):
    gpu_stats_dfs = []
    if hpc_log:
        make_round_gpu_metrics_dir(server_address, d_name, client_id, round_no)
    test_running_loss = 0.0

    test_labels = torch.FloatTensor().to(DEVICE)
    test_preds_prob = torch.FloatTensor().to(DEVICE)
    test_preds = torch.FloatTensor().to(DEVICE)
    model.eval()
    logger.info("Testing: ")
    with torch.no_grad():
        for batch_idx, (image, batch_label) in enumerate(test_loader):
            logits = model(image)
            loss = criterion(logits, batch_label)
            test_running_loss += loss.item()

            output = torch.sigmoid(logits)
            test_preds_prob = torch.cat((test_preds_prob, output.data), 0)
            pred = (output.data > 0.5).type(torch.float32)
            test_preds = torch.cat((test_preds, pred.data), 0)
            test_labels = torch.cat((test_labels, batch_label.data), 0)

            if batch_idx % 50 == 0:
                logger.info(f"batch_idx: {batch_idx}\n"
                            f"running_loss: {test_running_loss / (batch_idx + 1):.4f}\n")
                if hpc_log and batch_idx % 300 == 0:
                    gpu_stats_dfs.append(log_gpu_utilization_csv(d_name, client_id, round_no, 0, batch_idx))

    test_preds = test_preds.cpu().numpy().astype(np.int32)
    test_preds_prob = test_preds_prob.cpu().numpy()
    test_labels = test_labels.cpu().numpy().astype(np.int32)

    logger.info("Test report:")
    report = classification_report(test_labels, test_preds, target_names=classes_names)
    logger.info(report)
    report_json = json.dumps(
        classification_report(test_labels, test_preds, target_names=classes_names, output_dict=True))

    aucs = {}
    for i, c in enumerate(classes_names):
        # Do not calculate ROC AUC for classes with 0 samples
        positive_test_labels_count = sum(test_labels.astype(np.float32)[:, i] == 1.0)
        negative_test_labels_count = sum(test_labels.astype(np.float32)[:, i] != 1.0)
        if positive_test_labels_count > 0 and negative_test_labels_count > 0:
            aucs[c] = roc_auc_score(test_labels.astype(np.float32)[:, i], test_preds_prob[:, i])

    aucs_json = json.dumps(aucs)
    avg_auc = np.mean(list(aucs.values()))

    test_loss = test_running_loss / len(test_loader)
    logger.info(f" Test Loss: {test_loss:.4f}")
    logger.info(f" Test avg AUC: {avg_auc:.4f}")
    logger.info(f" Test AUCs: {aucs}")

    if hpc_log:
        save_round_gpu_csv(gpu_stats_dfs, server_address, d_name, str(client_id), round_no)

    return avg_auc, test_loss, report_json, aucs_json
