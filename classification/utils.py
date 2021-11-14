import torch
from collections import OrderedDict
from sklearn.metrics import classification_report
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def accuracy_score(pred, actual):
    act_labels = actual == 1
    same = act_labels == pred
    correct = same.sum().item()
    total = actual.shape[0] * actual.shape[1]
    return correct / total


def get_state_dict(model, parameters):
    params_dict = []
    for i, k in enumerate(list(model.state_dict().keys())):
        p = parameters[i]
        if 'num_batches_tracked' in k:
            p = p.reshape(p.size)
        params_dict.append((k, p))
    return OrderedDict({k: torch.Tensor(v) for k, v in params_dict})


def get_train_transformation_albu(height, width):
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
        albu.CoarseDropout(fill_value=0, p=0.25, max_height=32, max_width=32, max_holes=8),
        albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ToTensorV2(),
    ])


def get_train_transform_albu(height, width):
    return albu.Compose([
        albu.Resize(height, width),
        albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ToTensorV2(),
    ])


def test(model, device, logger, test_loader, criterion, optimizer, scheduler, classes_names):
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
            acc = accuracy_score(pred, label)

            test_preds.append(pred)
            test_labels.append(label)

            test_running_loss += loss.item()
            test_running_accuracy += acc

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
    report = classification_report(test_preds, test_labels, target_names=classes_names)
    logger.info(report)
    return test_acc, test_loss, report
