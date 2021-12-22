import os
import torch
from collections import OrderedDict
from sklearn.metrics import classification_report, accuracy_score
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
import torchvision.transforms.functional as F_vision
from PIL import Image

import numpy as np

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torchvision
import argparse
from collections import defaultdict, Counter

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NIH_DATASET_PATH_BASE = os.path.expandvars("$SCRATCH/fl_msc/classification/NIH/data/")
RSNA_DATASET_PATH_BASE = os.path.expandvars("$SCRATCH/fl_msc/classification/RSNA/")
# RSNA_DATASET_PATH_BASE = os.path.expandvars("/Users/filip/Data/Studies/MastersThesis/Datasets/RSNA-pneumonia-detection-challenge/kaggle")

COVID19_DATASET_PATH_BASE = os.path.expandvars(
    "$SCRATCH/fl_msc/classification/COVID-19_Radiography_Dataset/")


def accuracy_score_batch(pred, actual):
    act_labels = actual == 1
    same = act_labels == pred
    correct = same.sum().item()
    total = actual.shape[0] * actual.shape[1]
    return correct / total


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


def get_train_transformation_albu_NIH(height, width):
    return albu.Compose([
        albu.RandomResizedCrop(height=height, width=width, scale=(0.5, 1.0),
                               ratio=(0.8, 1.2), interpolation=1, p=1.0),
        albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, interpolation=1,
                              border_mode=0, value=0, p=0.25),
        albu.HorizontalFlip(p=0.5),
        albu.OneOf([
            albu.MotionBlur(p=.2),
            albu.MedianBlur(blur_limit=3, p=0.1),
            albu.Blur(blur_limit=3, p=0.1),
        ], p=0.25),
        albu.OneOf([
            albu.OpticalDistortion(p=0.3),
            albu.GridDistortion(p=0.3),
            albu.IAAPiecewiseAffine(p=0.1),
            albu.ElasticTransform(alpha=1, sigma=50, alpha_affine=40,
                                  interpolation=1, border_mode=4,  # value=None, mask_value=None,
                                  always_apply=False, approximate=False, p=0.3)
        ], p=0.3),
        albu.OneOf([
            albu.CLAHE(clip_limit=2),
            albu.IAASharpen(),
            albu.IAAEmboss(),
            albu.RandomBrightnessContrast(),
        ], p=0.25),
        # albu.CoarseDropout(fill_value=0, p=0.25, max_height=32, max_width=32, max_holes=8),
        albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ToTensorV2(),
    ])


def get_test_transform_albu_NIH(height, width):
    return albu.Compose([
        albu.Resize(height, width),
        albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ToTensorV2(),
    ])


def get_train_transform_covid_19_rd(args):
    return torchvision.transforms.Compose([
        # Converting images to the size that the model expects
        torchvision.transforms.Resize(size=(args.size, args.size)),
        torchvision.transforms.RandomHorizontalFlip(),  # A RandomHorizontalFlip to augment our data
        torchvision.transforms.ColorJitter(
            brightness=[0.8, 1.2],
            contrast=[0.8, 1.2],
            saturation=[0.8, 1.2],
            hue=[-0.1, 0.1]
        ),
        torchvision.transforms.RandomAffine(
            degrees=[-90, 90],
            translate=[0.2, 0.2],
            scale=[1, 1.3],
            shear=[-10, 10],
        ),
        torchvision.transforms.ToTensor(),  # Converting to tensor
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # Normalizing the data to the data that the ResNet34 was trained on

    ])


def get_test_transform_covid_19_rd(args):
    return torchvision.transforms.Compose([
        # Converting images to the size that the model expects
        torchvision.transforms.Resize(size=(args.size, args.size)),
        # We don't do data augmentation in the test/val set
        torchvision.transforms.ToTensor(),  # Converting to tensor
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # Normalizing the data to the data that the ResNet34 was trained on

    ])


def make_patch(args, segmentation_model, image, patient_id):
    image = image.convert("L")
    # image.save(f'/Users/filip/Data/Studies/MastersThesis/tmp_patches/L_{idx}.png', 'PNG')
    # image.save(f'/net/scratch/people/plgfilipsl/tmp_patches/L_{idx}.png', 'PNG')
    resize_transform = torchvision.transforms.Resize(size=(args.segmentation_size, args.segmentation_size))
    image = resize_transform(image)
    image = F_vision.to_tensor(image)
    image = image[None, ...]
    image = image.to(DEVICE)
    segmentation_model.eval()

    with torch.no_grad():
        outputs_mask = segmentation_model(image)

    image = image[0, 0, :]
    img_np = image.cpu().numpy()
    # Image.fromarray((img_np).astype(np.int8), mode='L').convert('RGB').save(f'/Users/filip/Data/Studies/MastersThesis/tmp_patches/img_np_{idx}.png', 'PNG')
    # Image.fromarray(img_np.astype(np.uint8), mode='L').convert('RGB').save(f'/net/scratch/people/plgfilipsl/tmp_patches/img_np_{idx}.png', 'PNG')

    out = outputs_mask[0, 0, :]
    out_np = out.cpu().numpy()
    # Image.fromarray((out_np*255).astype(np.int8)).convert('RGB').save(f'/Users/filip/Data/Studies/MastersThesis/tmp_patches/mask_np_{idx}.png', 'PNG')
    # Image.fromarray(out_np.astype(np.uint8), mode='L').convert('RGB').save(f'/net/scratch/people/plgfilipsl/tmp_patches/mask_np_{idx}.png', 'PNG')

    superposed = np.copy(img_np)
    superposed[out_np < 0.05] = 0
    return generate_patch(args, superposed, patient_id, args.size)


def trim_ranges(l, r, bound):
    if l < 0:
        r += abs(l)
        l = 0
    if r >= bound:
        l -= (bound - r)
        r = bound - 1
    return l, r


def generate_patch(args, masked_image, patient_id, patch_size=224):
    # Image.fromarray((255 * masked_image).astype(np.int8), mode='L').convert('RGB').save(
    #     os.path.join(RSNA_DATASET_PATH_BASE, f"masked_stage_2_train_images_{args.segmentation_size}/",
    #                  f"{patient_id}.png"), 'PNG')
    w, h = masked_image.shape
    shift = patch_size // 2

    x, y = np.where(masked_image > 0.5)
    x_filtered = []
    y_filtered = []

    for t in zip(x, y):
        if shift <= t[0] < args.segmentation_size - shift and shift <= t[1] < args.segmentation_size - shift:
            x_filtered.append(t[0])
            y_filtered.append(t[1])
    if len(x_filtered) == 0:
        x_filtered = [args.segmentation_size // 2]
        y_filtered = [args.segmentation_size // 2]
    i = np.random.randint(len(x_filtered))
    l_x, r_x = trim_ranges(x_filtered[i] - shift, x_filtered[i] + shift, w)
    l_y, r_y = trim_ranges(y_filtered[i] - shift, y_filtered[i] + shift, h)
    return Image.fromarray((255 * masked_image[l_x:r_x, l_y:r_y]).astype(np.int8), mode='L').convert('RGB')


def test_NIH(model, device, logger, test_loader, criterion, optimizer, scheduler, classes_names):
    test_running_loss = 0.0
    test_running_accuracy = 0.0
    test_preds = []
    test_labels = []
    model.eval()
    logger.info("Testing: ")
    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(test_loader):
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            output = model(image)
            output = torch.sigmoid(output)
            loss = criterion(output, label)
            pred = (output.data > 0.5).type(torch.float32)
            acc = accuracy_score_batch(pred, label)

            test_preds.append(pred)
            test_labels.append(label)

            test_running_loss += loss.item()
            test_running_accuracy += acc
            if batch_idx % 50 == 0:
                logger.info(f"batch_idx: {batch_idx}")

    test_loss = test_running_loss / len(test_loader)
    test_acc = test_running_accuracy / len(test_loader)

    scheduler.step(test_loss)

    for param_group in optimizer.param_groups:
        logger.info(f"Current lr: {param_group['lr']}")

    logger.info(f" Test Loss: {test_loss:.4f}"
                f" Test Acc: {test_acc:.4f}")

    test_preds = torch.cat(test_preds, dim=0).tolist()
    test_labels = torch.cat(test_labels, dim=0).tolist()
    logger.info("Test report:")
    report = classification_report(test_labels, test_preds, target_names=classes_names)
    logger.info(report)
    return test_acc, test_loss, report


def test_single_label(model, device, logger, test_loader, criterion, optimizer, classes_names):
    test_running_loss = 0.0
    test_running_accuracy = 0.0
    test_preds = []
    test_labels = []
    model.eval()
    logger.info("Testing: ")
    with torch.no_grad():
        for batch_idx, (image, batch_label) in enumerate(test_loader):
            image = image.to(device=device, dtype=torch.float32)
            batch_label = batch_label.to(device=device)

            logits = model(image)
            loss = criterion(logits, batch_label)

            test_running_loss += loss.item()
            test_running_accuracy += accuracy(logits, batch_label)

            y_pred = F.softmax(logits, dim=1)
            top_p, top_class = y_pred.topk(1, dim=1)

            test_labels.append(batch_label.view(*top_class.shape))
            test_preds.append(top_class)

            if batch_idx % 50 == 0:
                logger.info(f"batch_idx: {batch_idx}")

    test_loss = test_running_loss / len(test_loader)
    test_acc = test_running_accuracy / len(test_loader)

    for param_group in optimizer.param_groups:
        logger.info(f"Current lr: {param_group['lr']}")

    logger.info(f" Test Loss: {test_loss:.4f}"
                f" Test Acc: {test_acc:.4f}")

    test_preds = torch.cat(test_preds, dim=0).tolist()
    test_labels = torch.cat(test_labels, dim=0).tolist()
    logger.info("Test report:")
    report = classification_report(test_labels, test_preds, target_names=classes_names)
    logger.info(report)
    return test_acc, test_loss, report


def test_single_label_patching(model, device, logger, test_patching_dataset, criterion, optimizer, classes_names, K):
    test_running_loss = 0.0
    test_running_accuracy = 0.0
    test_preds = []
    test_labels = []

    model.eval()
    logger.info("Testing: ")
    test_patching_loader = torch.utils.data.DataLoader(test_patching_dataset, batch_size=1, num_workers=0)
    test_preds_per_idx = defaultdict(list)
    test_labels_per_idx = dict()

    with torch.no_grad():
        for i in range(K):
            logger.info(f"repetition: {i}")
            for image_idx, (image, batch_label) in enumerate(test_patching_loader):
                image = image.to(device=device, dtype=torch.float32)
                batch_label = batch_label.to(device=device)

                logits = model(image)
                loss = criterion(logits, batch_label)

                test_running_loss += loss.item()
                test_running_accuracy += accuracy(logits, batch_label)

                y_pred = F.softmax(logits, dim=1)
                _, top_class = y_pred.topk(1, dim=1)

                test_labels_per_idx[image_idx] = batch_label.tolist()[0]
                test_preds_per_idx[image_idx] = test_preds_per_idx[image_idx] + top_class.tolist()[0]

                test_labels.append(batch_label.view(*top_class.shape))
                test_preds.append(top_class)

                if image_idx % 100 == 0:
                    logger.info(f"repetition: {i} image_idx: {image_idx}, ")

    test_loss = test_running_loss / (len(test_patching_loader) * K)
    test_patches_acc = test_running_accuracy / (len(test_patching_loader) * K)

    for param_group in optimizer.param_groups:
        logger.info(f"Current lr: {param_group['lr']}")

    logger.info(f" Test Patches Loss: {test_loss:.4f}"
                f" Test Patches Acc: {test_patches_acc:.4f}")

    test_labels = [test_labels_per_idx[i] for i in range(len(test_patching_loader))]
    test_preds = [Counter(test_preds_per_idx[i]).most_common(n=1)[0][0] for i in range(len(test_patching_loader))]

    test_acc = accuracy_score(test_preds, test_labels)

    logger.info("Test report (with majority voting):")
    report_majority_voting = classification_report(test_labels, test_preds, target_names=classes_names)
    logger.info(report_majority_voting)

    return test_acc, test_loss, report_majority_voting


def parse_args():
    parser = argparse.ArgumentParser(description="Train classifier to detect covid on CXR images.")

    parser.add_argument("--images",
                        type=str,
                        default=os.path.join(RSNA_DATASET_PATH_BASE, "masked_stage_2_train_images/"),
                        # default=os.path.join(RSNA_DATASET_PATH_BASE, "stage_2_train_images/"),
                        help="Path to the images")
    parser.add_argument("--labels",
                        type=str,
                        default=os.path.join(RSNA_DATASET_PATH_BASE, "nih_data_labels.csv"),
                        help="Path to the labels")
    parser.add_argument("--train_subset",
                        type=str,
                        default=os.path.join(RSNA_DATASET_PATH_BASE, "train_labels_stage_1.csv"),
                        help="Path to the file with training/validation dataset files list")
    parser.add_argument("--test_subset",
                        type=str,
                        default=os.path.join(RSNA_DATASET_PATH_BASE, "test_labels_stage_1.csv"),
                        help="Path to the file with test dataset files list")
    parser.add_argument("--segmentation_model",
                        type=str,
                        # default="/net/archive/groups/plggsano/fl_msc/unet_model",
                        # default="/Users/filip/Data/Studies/MastersThesis/unet_model",
                        default="/net/archive/groups/plggsano/fl_msc/unet_14_jacc_0.906_loss_0.15",
                        help="Path to the file with segmentation model")
    parser.add_argument("--patches",
                        type=bool,
                        default=True,
                        help="whether to train model utilizing patching approach")
    parser.add_argument("--k_patches_client",
                        type=int,
                        default=1,
                        help="number of patches generated for an image in client")
    parser.add_argument("--k_patches_server",
                        type=int,
                        default=10,
                        # default=1,
                        help="number of patches generated for an image in server")
    parser.add_argument("--in_channels",
                        type=int,
                        default=3,
                        help="Number of input channels")
    parser.add_argument("--local_epochs",
                        type=int,
                        default=1,
                        help="Number of local epochs")
    parser.add_argument("--segmentation_size",
                        type=int,
                        default=1024,
                        help="input image size in segmentation model")
    parser.add_argument("--size",
                        type=int,
                        default=224,
                        help="input image size in classification model")
    parser.add_argument("--num_workers",
                        type=int,
                        default=0,
                        help="Number of workers for processing the data")
    parser.add_argument("--classes",
                        type=int,
                        default=3,
                        help="Number of classes in the dataset")
    parser.add_argument("--batch_size",
                        type=int,
                        default=8,
                        help="Number of batch size")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-3,
                        help="Number of learning rate")
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.0,
                        help="Number of weight decay")
    parser.add_argument("--device_id",
                        type=str,
                        default="0",
                        help="GPU ID")
    parser.add_argument("--limit",
                        type=int,
                        default=-1,
                        help="use to limit amount of data")

    parser.add_argument("--node_name",
                        type=str,
                        default="p001",
                        help="server node name")

    parser.add_argument("--client_id",
                        type=int,
                        default=0,
                        help="ID of the client")

    parser.add_argument("--clients_number",
                        type=int,
                        default=3,
                        help="number of the clients")

    parser.add_argument("--dataset",
                        type=str,
                        default="chest",
                        help="kind of dataset: chest/mnist")

    parser.add_argument("--num_rounds",
                        type=int,
                        default=100,
                        help="number of rounds")

    args = parser.parse_args()
    return args
