import pandas as pd
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from data_selector import IIDSelector
from utils import NIH_CHESTDX_CLASSES


class NIHDataset(Dataset):
    def __init__(self, client_id, clients_number, dataset_split_file, labels, images_dir, limit=-1):
        # Order of classes adjusted to ChestDx dataset
        self.classes_names = NIH_CHESTDX_CLASSES
        print(os.getcwd())
        with open(dataset_split_file, "r") as file:
            file_content = file.readlines()

        file_content = [i.rstrip('\n') for i in file_content]
        files = pd.DataFrame(file_content, columns=['file'])
        self.images = [os.path.join(images_dir, file) for file in file_content]

        # LABELS
        labels_nih = pd.read_csv(labels)
        subset_labels = files.merge(labels_nih, left_on="file", right_on="Image Index")
        self.one_hot_labels = []
        self.one_hot_multiclass(subset_labels["Finding Labels"].tolist())

        if limit != -1:
            self.images = self.images[0:limit]
            self.one_hot_labels = self.one_hot_labels[0:limit]

        selector = IIDSelector()
        if client_id != -1:
            self.images, self.one_hot_labels = selector.select_data(self.images, self.one_hot_labels, client_id,
                                                                    clients_number)
        self.images_count = len(self.images)

    def __len__(self):
        return self.images_count

    def __getitem__(self, idx):
        path = self.images[idx]
        pil_image = Image.open(path).convert('RGB')
        label = self.one_hot_labels[idx]
        return pil_image, label

    def one_hot_multiclass(self, labels_list):
        for elem in labels_list:
            elem = elem.split('|')
            labels_vector = np.zeros(len(self.classes_names), dtype=np.float32)
            for idx, item in enumerate(self.classes_names):
                if item in elem:
                    labels_vector[idx] = 1.0
            self.one_hot_labels.append(labels_vector)
