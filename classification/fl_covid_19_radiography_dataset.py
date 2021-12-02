import glob
import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
import numpy as np
import cv2
from data_selector import IIDSelector
import pydicom
import skimage.io
import skimage.transform
from skimage.transform import SimilarityTransform, AffineTransform
import math
from imgaug import augmenters as iaa
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Covid19RDDataset(Dataset):
    def __init__(self, client_id, clients_number, ids_labels_file, images_source, transform=None, debug=False,
                 limit=-1):
        super(Covid19RDDataset, self).__init__()

        self.debug = debug
        self.transform = transform

        # "Normal"=0, "COVID"=1
        self.classes_names = ["Normal", "COVID"]

        ids_labels_df = pd.read_csv(ids_labels_file)

        if self.debug:
            samples = ids_labels_df.head(32)
            print("Debug mode, samples: ", samples)

        # IMAGES
        self.images = [os.path.join(images_source, row['file_name']) for _, row in ids_labels_df.iterrows()]
        images_count = len(self.images)
        print(f'Dataset file:{ids_labels_file}, len = {images_count}')

        # LABELS
        self.labels = [row['label'] for _, row in ids_labels_df.iterrows()]

        if limit != -1:
            self.images = self.images[0:limit]
            self.labels = self.labels[0:limit]

        selector = IIDSelector()
        if client_id != -1:
            self.images, self.labels = selector.select_data(self.images, self.labels, client_id,
                                                            clients_number)

    def __len__(self):
        return len(self.labels)

    def num_classes(self):
        return 2

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        return self.transform(image), self.labels[idx]
