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


class RSNADataset(Dataset):
    def __init__(self, client_id, clients_number, ids_labels_file, img_size, images_source,
                 augmentation_level=10, crop_source=1024, is_training=True, debug=False, limit=-1):
        super(RSNADataset, self).__init__()

        self.is_training = is_training
        self.img_size = img_size
        self.debug = debug
        self.crop_source = crop_source
        self.augmentation_level = augmentation_level

        # Target = 1 => Pneumonia, Target = 0 => No pneumonia
        self.classes_names = ["No pneumonia", "Pneumonia"]

        ids_labels_df = pd.read_csv(ids_labels_file)

        if self.debug:
            samples = ids_labels_df.head(32)
            print("Debug mode, samples: ", samples)

        # IMAGES
        self.images = [os.path.join(images_source, row['patient_id']) + '.dcm' for _, row in ids_labels_df.iterrows()]
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

    def get_image(self, img_path):
        """Load a dicom image to an array"""
        try:
            dcm_data = pydicom.read_file(img_path)
            img = dcm_data.pixel_array
            return img
        except:
            pass

    def num_classes(self):
        return 2

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = self.get_image(img_path)
        label = self.labels[idx]

        if self.crop_source != 1024:
            img_source_w = self.crop_source
            img_source_h = self.crop_source
        else:
            img_source_h, img_source_w = img.shape[:2]
        img_h, img_w = img.shape[:2]

        # set augmentation levels
        augmentation_sigma = {
            1: dict(scale=0, angle=0, shear=0, gamma=0, hflip=False),
            10: dict(scale=0.1, angle=5.0, shear=2.5, gamma=0.2, hflip=False),
            15: dict(scale=0.15, angle=6.0, shear=4.0, gamma=0.2, hflip=np.random.choice([True, False])),
            20: dict(scale=0.15, angle=6.0, shear=4.0, gamma=0.25, hflip=np.random.choice([True, False])),
        }[self.augmentation_level]
        # training mode augments
        if self.is_training:
            cfg = TransformCfg(
                crop_size=self.img_size,
                src_center_x=img_w / 2 + np.random.uniform(-32, 32),
                src_center_y=img_h / 2 + np.random.uniform(-32, 32),
                scale_x=self.img_size / img_source_w * (2 ** np.random.normal(0, augmentation_sigma["scale"])),
                scale_y=self.img_size / img_source_h * (2 ** np.random.normal(0, augmentation_sigma["scale"])),
                angle=np.random.normal(0, augmentation_sigma["angle"]),
                shear=np.random.normal(0, augmentation_sigma["shear"]),
                hflip=augmentation_sigma["hflip"],
                vflip=False,
            )
        # validation mode augments
        else:
            cfg = TransformCfg(
                crop_size=self.img_size,
                src_center_x=img_w / 2,
                src_center_y=img_h / 2,
                scale_x=self.img_size / img_source_w,
                scale_y=self.img_size / img_source_h,
                angle=0,
                shear=0,
                hflip=False,
                vflip=False,
            )
        # add more augs in training modes
        crop = cfg.transform_image(img)
        if self.is_training:
            crop = np.power(crop, 2.0 ** np.random.normal(0, augmentation_sigma["gamma"]))
            if self.augmentation_level == 20:
                aug = iaa.Sequential(
                    [
                        iaa.Sometimes(0.1, iaa.CoarseSaltAndPepper(p=(0.01, 0.01), size_percent=(0.1, 0.2))),
                        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0.0, 2.0))),
                        iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 0.04 * 255))),
                    ]
                )
                crop = (
                        aug.augment_image(np.clip(np.stack([crop, crop, crop], axis=2) * 255, 0, 255).astype(np.uint8))[
                        :, :, 0].astype(np.float32)
                        / 255.0
                )
            if self.augmentation_level == 15:
                aug = iaa.Sequential(
                    [iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0.0, 1.0))),
                     iaa.Sometimes(0.25, iaa.AdditiveGaussianNoise(scale=(0, 0.02 * 255)))]
                )
                crop = (
                        aug.augment_image(np.clip(np.stack([crop, crop, crop], axis=2) * 255, 0, 255).astype(np.uint8))[
                        :, :, 0].astype(np.float32)
                        / 255.0
                )
        im = Image.fromarray(crop).convert('RGB')
        im_array = np.array(im)
        transform_norm = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
        img_normalized = transform_norm(im_array)
        return np.array(img_normalized), label


class TransformCfg:
    """
    Configuration structure for crop parameters
    and augmentations
    """

    def __init__(self, crop_size: int, src_center_x: int, src_center_y: int, scale_x: float = 1.0, scale_y: float = 1.0,
                 angle: float = 0.0, shear: float = 0.0, hflip: bool = False, vflip: bool = False):
        self.crop_size = crop_size
        self.src_center_x = src_center_x
        self.src_center_y = src_center_y
        self.angle = angle
        self.shear = shear
        self.scale_y = scale_y
        self.scale_x = scale_x
        self.vflip = vflip
        self.hflip = hflip

    def __str__(self) -> str:
        return str(self.__dict__)

    def transform(self) -> AffineTransform:
        scale_x = self.scale_x
        if self.hflip:
            scale_x *= -1
        scale_y = self.scale_y
        if self.vflip:
            scale_y *= -1

        tform = skimage.transform.AffineTransform(translation=(self.src_center_x, self.src_center_y))
        tform = skimage.transform.AffineTransform(scale=(1.0 / self.scale_x, 1.0 / self.scale_y)) + tform
        tform = skimage.transform.AffineTransform(rotation=self.angle * math.pi / 180,
                                                  shear=self.shear * math.pi / 180) + tform
        tform = skimage.transform.AffineTransform(translation=(-self.crop_size / 2, -self.crop_size / 2)) + tform

        return tform

    def transform_image(self, img: np.array) -> np.array:
        crop = skimage.transform.warp(img, self.transform(), mode="constant", cval=0, order=1,
                                      output_shape=(self.crop_size, self.crop_size))
        return crop
