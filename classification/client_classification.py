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

from fl_mnist_dataset import MNISTDataset
from utils import get_state_dict, get_train_transformation_albu, accuracy_score, test, \
    get_test_transform_albu, get_ENS_weights, parse_args

hdlr = logging.StreamHandler()
logger = logging.getLogger(__name__)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.info(f"Device: {device}")
logger.info("CUDA_VISIBLE_DEVICES =" + str(os.environ['CUDA_VISIBLE_DEVICES']))

args = parse_args()


# os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id


def train(model, train_loader, criterion, optimizer, classes_names, epochs=1):
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


def load_data(client_id, clients_number):
    train_transform_albu = get_train_transformation_albu(args.size, args.size)
    test_transform_albu = get_test_transform_albu(args.size, args.size)

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


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

def main():
    """Create model, load data, define Flower client, start Flower client."""

    server_addr = args.node_name
    client_id = args.client_id
    clients_number = args.clients_number
    # Load model
    # EFFNET
    model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=args.classes, in_channels=args.in_channels)
    model.cuda()

    # Load data
    train_loader, test_loader, one_hot_labels, classes_names = load_data(client_id, clients_number)

    ens = get_ENS_weights(args.classes, list(sum(one_hot_labels)), beta=args.beta)
    ens /= ens.max()
    logger.info(f"beta: {args.beta}, weights: {ens.tolist()}")
    ens = ens.to(device=device, dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(weight=ens).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, min_lr=1e-6)

    # Flower client
    class ClassificationClient(fl.client.NumPyClient):
        def get_parameters(self):
            return [val.cpu().numpy() for _, val in model.state_dict().items()]

        def set_parameters(self, parameters):
            logger.info("Loading parameters...")
            state_dict = get_state_dict(model, parameters)
            model.load_state_dict(state_dict, strict=True)
            logger.info("Parameters loaded")

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            train(model, train_loader, criterion, optimizer, classes_names, epochs=args.local_epochs)
            return self.get_parameters(), len(train_loader), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy, report = test(model, test_loader, device, logger, criterion, optimizer, scheduler,
                                          classes_names)
            logger.info(f"Loss: {loss}, accuracy: {accuracy}")
            logger.info(report)
            return float(loss), len(test_loader), {"accuracy": float(accuracy), "loss": float(loss)}

    # Start client
    logger.info("Connecting to:" + f"{server_addr}:8081")
    fl.client.start_numpy_client(f"{server_addr}:8081", client=ClassificationClient())


if __name__ == "__main__":
    main()
