from abc import ABC, abstractmethod


class DataSelector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def select_data(self, images, masks, client_id, number_of_clients):
        pass


class IIDSelector(DataSelector):
    def select_data(self, images, masks, client_id, number_of_clients):
        sampled_images = [path for i, path in enumerate(images) if i % client_id == 0]
        sampled_masks = [path for i, path in enumerate(masks) if i % client_id == 0]
        return sampled_images, sampled_masks
