import os
import torch
from collections import OrderedDict
from sklearn.metrics import classification_report, accuracy_score

import torch.nn.functional as F

import torchvision
import json

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

REPO_DATASETS_PATH_BASE = os.path.expandvars("$SCRATCH/FederatedLearning_MSc/classification/datasets")
RSNA_DATASET_PATH_BASE = os.path.expandvars("$SCRATCH/fl_msc/classification/RSNA/")
NIH_DATASET_PATH_BASE = os.path.expandvars("$SCRATCH/fl_msc/classification/NIH/")
CHESTDX_DATASET_PATH_BASE = os.path.expandvars(
    "$PLG_GROUPS_STORAGE/plggsano/fl_msc_classification/classification/China_X_ray")
CHEXPERT_DATASET_PATH_BASE = os.path.expandvars("$PLG_GROUPS_STORAGE/plggsano/Chexpert/Chexpert_dataset")
MIMIC_DATASET_PATH_BASE = os.path.expandvars(
    "$SCRATCH/fl_msc/classification/MIMIC/mimic-cxr-jpg-2.0.0.physionet.org/files")


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
        model.fc = torch.nn.Linear(in_features=2048, out_features=classes)
        model = model.to(DEVICE)
        return model
    if m == 'DenseNet121':
        model = torchvision.models.densenet121(pretrained=True)
        model.classifier = torch.nn.Linear(in_features=1024, out_features=classes)
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
        return images_dir, train_subset, test_subset
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
        return images_dir, train_subset, test_subset
    elif dataset == 'chestdx-pe':
        images_dir = CHESTDX_DATASET_PATH_BASE
        test_subset = os.path.join(REPO_DATASETS_PATH_BASE, 'ChestDx-PE/full.csv')
        return images_dir, test_subset
    elif dataset == 'cc-cxri-p':
        images_dir = CHESTDX_DATASET_PATH_BASE
        train_subset = os.path.join(REPO_DATASETS_PATH_BASE, 'CC-CXRI-P/train.csv')
        test_subset = os.path.join(REPO_DATASETS_PATH_BASE, 'CC-CXRI-P/test.csv')
        return images_dir, train_subset, test_subset
    elif dataset == 'chexpert':
        images_dir = CHEXPERT_DATASET_PATH_BASE
        train_subset = os.path.join(REPO_DATASETS_PATH_BASE, 'CheXpert/train.csv')
        test_subset = os.path.join(REPO_DATASETS_PATH_BASE, 'CheXpert/valid.csv')
        return images_dir, train_subset, test_subset
    elif dataset == 'mimic':
        images_dir = MIMIC_DATASET_PATH_BASE
        train_subset = os.path.join(REPO_DATASETS_PATH_BASE, 'MIMIC/merged_train.csv')
        test_subset = os.path.join(REPO_DATASETS_PATH_BASE, 'MIMIC/merged_test.csv')
        return images_dir, train_subset, test_subset


def get_beton_data_paths(dataset):
    if dataset == 'rsna':
        train_subset = os.path.join(RSNA_DATASET_PATH_BASE, 'train-jpg90.beton')
        test_subset = os.path.join(RSNA_DATASET_PATH_BASE, 'test-jpg90.beton')
        return train_subset, test_subset
    if dataset == 'nih':
        train_subset = os.path.join(NIH_DATASET_PATH_BASE, 'nih-train-jpg90.beton')
        test_subset = os.path.join(NIH_DATASET_PATH_BASE, 'nih-test-jpg90.beton')
        return train_subset, test_subset


def test_single_label(model, device, logger, test_loader, criterion, classes_names):
    test_running_loss = 0.0
    test_running_accuracy = 0.0
    test_preds = []
    test_labels = []
    model.eval()
    logger.info("Testing: ")
    with torch.no_grad():
        for batch_idx, (image, batch_label) in enumerate(test_loader):
            # image = image.to(device=device, dtype=torch.float32)
            # batch_label = batch_label.to(device=device)
            batch_label = torch.flatten(batch_label)

            logits = model(image)
            loss = criterion(logits, batch_label)

            test_running_loss += loss.item()
            test_running_accuracy += accuracy(logits, batch_label)

            y_pred = F.softmax(logits, dim=1)
            top_p, top_class = y_pred.topk(1, dim=1)

            test_labels.append(batch_label.view(*top_class.shape))
            test_preds.append(top_class)

            if batch_idx % 50 == 0:
                logger.info(f"batch_idx: {batch_idx}\n"
                            f"running_loss: {test_running_loss / (batch_idx + 1):.4f}\n"
                            f"running_acc: {test_running_accuracy / (batch_idx + 1):.4f}\n\n")

    test_preds = torch.cat(test_preds, dim=0).tolist()
    test_labels = torch.cat(test_labels, dim=0).tolist()
    logger.info("Test report:")
    report = classification_report(test_labels, test_preds, target_names=classes_names)
    logger.info(report)
    report_json = json.dumps(
        classification_report(test_labels, test_preds, target_names=classes_names, output_dict=True))

    test_loss = test_running_loss / len(test_loader)
    test_acc = accuracy_score(test_labels, test_preds)
    logger.info(f" Test Loss: {test_loss:.4f}"
                f" Test Acc: {test_acc:.4f}")

    return test_acc, test_loss, report_json
