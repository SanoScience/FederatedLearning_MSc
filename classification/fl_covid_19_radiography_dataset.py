import pandas as pd
import os
from torch.utils.data import Dataset
from PIL import Image, ImageFile

from data_selector import IIDSelector
from utils import make_patch

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Covid19RDDataset(Dataset):
    def __init__(self, args, client_id, clients_number, ids_labels_file, images_source, transform=None, debug=False,
                 limit=-1, segmentation_model=None):
        super(Covid19RDDataset, self).__init__()

        self.args = args
        self.debug = debug
        self.transform = transform
        self.segmentation_model = segmentation_model

        # "Normal"=0, "COVID"=1, "Lung_Opacity"=2, "Viral Pneumonia"=3
        self.classes_names = ["Normal", "COVID", "Lung_Opacity", "Viral Pneumonia"]

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
        return self.args.classes

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        if self.segmentation_model:
            image = make_patch(self.args, self.segmentation_model, image)
        return self.transform(image), self.labels[idx]
