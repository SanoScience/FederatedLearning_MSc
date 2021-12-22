import random
from abc import ABC, abstractmethod
from collections import defaultdict
import os


class DataSelector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def select_client_data(self, images, masks, client_id, number_of_clients, labels_dict):
        pass

    @abstractmethod
    def select_server_data(self, images, masks, labels_dict):
        pass


class IIDSelector(DataSelector):
    def prepare_order(self, images, masks, labels_dict):
        data = defaultdict(list)
        image_to_mask = {}
        for i, m in zip(images, masks):
            image_to_mask[os.path.basename(i)] = (i, m)
        for k, v in labels_dict.items():
            # data is mapping from class to filename
            # print(k, v)
            data[v].append(k)
        res_images = []
        res_masks = []

        for cls, imgs in data.items():
            for img in imgs:
                if img in image_to_mask and cls != 'no_class':
                    res_img, res_mask = image_to_mask[img]
                    res_images.append(res_img)
                    res_masks.append(res_mask)

        return res_images, res_masks

    def select_client_data(self, images, masks, client_id, number_of_clients, labels_dict):
        n = len(images)
        ordered_images, ordered_masks = self.prepare_order(images, masks, labels_dict)
        test_fraction = int(0.9 * n)
        client_images = ordered_images[:test_fraction]
        client_masks = ordered_masks[:test_fraction]
        sampled_images = [path for i, path in enumerate(client_images) if (i % number_of_clients) == client_id]
        sampled_masks = [path for i, path in enumerate(client_masks) if (i % number_of_clients) == client_id]
        zipped = list(zip(sampled_images, sampled_masks))
        random.shuffle(zipped)
        sampled_images, sampled_masks = zip(*zipped)
        return sampled_images, sampled_masks

    def select_server_data(self, images, masks, labels_dict):
        n = len(images)
        ordered_images, ordered_masks = self.prepare_order(images, masks, labels_dict)
        test_fraction = int(0.9 * n)
        return ordered_images[test_fraction:], ordered_masks[test_fraction:]
