import pandas as pd

import os
from torch.utils.data import Dataset
from PIL import Image, ImageFile

import pydicom

import torch
import torchvision

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ImageFile.LOAD_TRUNCATED_IMAGES = True

RSNA_DATASET_PATH_BASE = os.path.expandvars("$SCRATCH/fl_msc/classification/RSNA/")
IMAGES_DIR = os.path.join(RSNA_DATASET_PATH_BASE, "stage_2_train_images/")
LABELS = os.path.join(RSNA_DATASET_PATH_BASE, "test_labels_stage_1.csv")

max_size = 512, 512


class RSNADataset(Dataset):
    def __init__(self, ids_labels_file, images_source):
        super(RSNADataset, self).__init__()

        # "Normal"=0, "No Lung Opacity / Not Normal"=1, "Lung Opacity"=2
        self.classes_names = ["Normal", "No Lung Opacity / Not Normal", "Lung Opacity"]

        self.ids_labels_df = pd.read_csv(ids_labels_file)

        extension = '.dcm'

        # IMAGES
        self.images = [os.path.join(images_source, row['patient_id']) + extension for _, row in
                       self.ids_labels_df.iterrows()]
        images_count = len(self.images)
        print(f'Dataset file:{ids_labels_file}, len = {images_count}')

        # LABELS
        self.labels = [row['label'] for _, row in self.ids_labels_df.iterrows()]

    def __len__(self):
        return len(self.labels)

    def get_image(self, img_path):
        """Load a dicom image to an array"""
        try:
            dcm_data = pydicom.read_file(img_path)
            img = dcm_data.pixel_array
            return img
        except:
            pass

    def __getitem__(self, idx):
        image_path = self.images[idx]
        im_array = self.get_image(image_path)
        image_rgb = Image.fromarray(im_array).convert('RGB')
        patient_id = self.ids_labels_df.iloc[idx]['patient_id']
        image_rgb.thumbnail(max_size, Image.ANTIALIAS)
        image_rgb.save(
            f'/net/scratch/people/plgfilipsl/fl_msc/classification/RSNA/stage_2_train_images_jpg/{patient_id}.jpg',
            quality=90, subsampling=0)
        trans = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        return trans(image_rgb), self.labels[idx]


rsna_dataset = RSNADataset(LABELS, IMAGES_DIR)

test_patching_loader = torch.utils.data.DataLoader(rsna_dataset, batch_size=8, num_workers=36, pin_memory=True)

for image_idx, (image, batch_label) in enumerate(test_patching_loader):
    if image_idx % 50 == 0:
        print(f"batch_idx: {image_idx}")
