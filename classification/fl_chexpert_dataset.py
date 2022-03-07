import pandas as pd
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from data_selector import IIDSelector


class CheXpertDataset(Dataset):
    def __init__(self, client_id, clients_number, images_file, images_dir, limit=-1):
        # Order of classes and names the same as in MIMIC dataset
        self.classes_names = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
                              "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion",
                              "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"]

        self.images_file_df = pd.read_csv(images_file)
        self.images = [os.path.join(images_dir, row['Path']) for _, row in self.images_file_df.iterrows()]
        self.one_hot_labels = self.get_one_hot_multiclass(self.images_file_df)

        if limit != -1:
            self.images = self.images[:limit]
            self.one_hot_labels = self.one_hot_labels[:limit]

        self.images_count = len(self.images)
        selector = IIDSelector()
        if client_id != -1:
            self.images, self.one_hot_labels = selector.select_data(self.images, self.one_hot_labels, client_id,
                                                                    clients_number)

    def __len__(self):
        return self.images_count

    def __getitem__(self, idx):
        path = self.images[idx]
        image = Image.open(path).convert('RGB')
        label = self.one_hot_labels[idx]
        return image, label

    def get_one_hot_multiclass(self, labels_df):
        one_hot_labels = []
        for _, row in labels_df.iterrows():
            labels_vector = np.zeros(len(self.classes_names), dtype=np.float32)
            for idx, c_name in enumerate(self.classes_names):
                labels_vector[idx] = np.float32(row[c_name])
            one_hot_labels.append(labels_vector)
        return one_hot_labels
