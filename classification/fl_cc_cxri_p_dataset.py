import pandas as pd

import os
from torch.utils.data import Dataset
from PIL import Image

from data_selector import IIDSelector
from utils import CC_CXRI_P_CLASSES


class CCCXRIPDataset(Dataset):
    def __init__(self, client_id, clients_number, images_file, images_dir, transform, limit=-1):
        super(CCCXRIPDataset, self).__init__()

        # "Normal"=0, "Viral"=1, "COVID"=2, "Other"=3
        self.classes_names = CC_CXRI_P_CLASSES
        self.images_file_df = pd.read_csv(images_file)
        self.transform = transform

        # IMAGES
        self.images = [os.path.join(images_dir, row['path']) for _, row in self.images_file_df.iterrows()]
        # LABELS
        self.labels = [row['class'] for _, row in self.images_file_df.iterrows()]

        if limit != -1:
            self.images = self.images[:limit]
            self.labels = self.labels[:limit]

        selector = IIDSelector()
        if client_id != -1:
            self.images, self.labels = selector.select_data(self.images, self.labels, client_id, clients_number)
        self.images_count = len(self.images)
        print(f'Dataset file:{images_file}, len = {self.images_count}')

    def __len__(self):
        return self.images_count

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), self.classes_names.index(self.labels[idx])
