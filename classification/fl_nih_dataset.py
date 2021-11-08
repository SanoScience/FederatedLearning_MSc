import glob
import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
import numpy as np
import cv2
from data_selector import IIDSelector

ImageFile.LOAD_TRUNCATED_IMAGES = True


class NIHDataset(Dataset):
    def __init__(self, client_id, clients_number, dataset_split_file, labels, images_source, transform=None, limit=-1):

        self.transform = transform
        self.classes_names = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
                              "Pneumothorax", "Consolidation", "Edema", "Emphysema",
                              "Fibrosis", "Pleural_Thickening", "Hernia", "No Finding"]

        # IMAGES
        file = open(dataset_split_file, "r")
        file_content = file.readlines()
        file.close()
        file_content = [i.rstrip('\n') for i in file_content]
        files = pd.DataFrame(file_content, columns=['file'])
        self.images = [os.path.join(images_source, file) for file in file_content]
        images_count = len(self.images)

        # LABELS
        labels_nih = pd.read_csv(labels)
        subset_labels = files.merge(labels_nih, left_on="file", right_on="Image Index")
        self.one_hot_labels = []
        self.one_hot_multiclass(subset_labels["Finding Labels"].tolist())

        if limit != -1:
            self.images = self.images[0:limit]
            self.one_hot_labels = self.one_hot_labels[0:limit]

        selector = IIDSelector()
        self.images, self.one_hot_labels = selector.select_data(self.images, self.one_hot_labels, client_id,
                                                                clients_number)

    def __len__(self):
        return len(self.one_hot_labels)

    def __getitem__(self, idx):

        path = self.images[idx]

        # solve truncated image problem
        pil_image = Image.open(path).convert('RGB')
        open_cv_image = np.array(pil_image)
        # RGB to BGR
        image = open_cv_image[:, :, ::-1].copy()
        label = self.one_hot_labels[idx]

        if self.transform:
            image = self.transform(image=image)['image']

        return image, label

    def one_hot_multiclass(self, labels_list):
        for elem in labels_list:
            elem = elem.split('|')
            labels_vector = np.zeros(len(self.classes_names), dtype=np.long)
            for idx, item in enumerate(self.classes_names):
                if item in elem:
                    labels_vector[idx] = 1.0
            self.one_hot_labels.append(labels_vector)
