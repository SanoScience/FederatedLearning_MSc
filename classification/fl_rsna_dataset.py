import pandas as pd

import os
from torch.utils.data import Dataset
from PIL import Image, ImageFile

from utils import make_patch, generate_patch
from data_selector import IIDSelector
import pydicom
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True


class RSNADataset(Dataset):
    def __init__(self, args, client_id, clients_number, ids_labels_file, images_source, transform=None, debug=False,
                 limit=-1, segmentation_model=None):
        super(RSNADataset, self).__init__()

        self.args = args
        self.debug = debug
        self.transform = transform
        self.segmentation_model = segmentation_model

        # "Normal"=0, "No Lung Opacity / Not Normal"=1, "Lung Opacity"=2
        self.classes_names = ["Normal", "No Lung Opacity / Not Normal", "Lung Opacity"]

        self.ids_labels_df = pd.read_csv(ids_labels_file)

        if self.debug:
            samples = self.ids_labels_df.head(32)
            print("Debug mode, samples: ", samples)

        extension = '.png'

        # IMAGES
        self.images = [os.path.join(images_source, row['patient_id']) + extension for _, row in
                       self.ids_labels_df.iterrows()]
        images_count = len(self.images)
        print(f'Dataset file:{ids_labels_file}, len = {images_count}')

        # LABELS
        self.labels = [row['label'] for _, row in self.ids_labels_df.iterrows()]

        if limit != -1:
            self.images = self.images[:limit]
            self.labels = self.labels[:limit]

        selector = IIDSelector()
        if client_id != -1:
            self.images, self.labels = selector.select_data(self.images, self.labels, client_id,
                                                            clients_number)

    def __len__(self):
        return len(self.labels)

    def num_classes(self):
        return self.args.classes

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
        # im_array = self.get_image(image_path)
        # image_rgb = Image.fromarray(im_array).convert('RGB')
        image = Image.open(image_path).convert('L')
        image_l = np.array(image) / 255
        image_rgb = None
        if self.args.patches:
            # image_rgb.save(f'/net/scratch/people/plgfilipsl/tmp_patches/original_{idx}.png', 'PNG')
            # image_rgb.save(f'/Users/filip/Data/Studies/MastersThesis/tmp_patches/original_{idx}.png', 'PNG')
            # image_rgb = make_patch(self.args, self.segmentation_model, image_rgb,
            #                        self.ids_labels_df.iloc[idx]['patient_id'])
            image_rgb = generate_patch(self.args, image_l, self.ids_labels_df.iloc[idx]['patient_id'], patch_size=self.args.size)
            # image_rgb.save(f'/net/scratch/people/plgfilipsl/tmp_patches/{idx}.png', 'PNG')
            # image_rgb.save(f'/Users/filip/Data/Studies/MastersThesis/tmp_patches/{idx}.png', 'PNG')
        return self.transform(image_rgb), self.labels[idx]
