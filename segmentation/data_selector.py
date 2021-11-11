from abc import ABC, abstractmethod


class DataSelector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def select_client_data(self, images, masks, client_id, number_of_clients):
        pass

    @abstractmethod
    def select_server_data(self, images, masks):
        pass


class IIDSelector(DataSelector):
    def select_client_data(self, images, masks, client_id, number_of_clients):
        n = len(images)
        test_fraction = int(0.9 * n)
        client_images = images[:test_fraction]
        client_masks = masks[:test_fraction]
        sampled_images = [path for i, path in enumerate(client_images) if (i % number_of_clients) == client_id]
        sampled_masks = [path for i, path in enumerate(client_masks) if (i % number_of_clients) == client_id]
        return sampled_images[:5], sampled_masks[:5]

    def select_server_data(self, images, masks):
        n = len(images)
        test_fraction = int(0.9 * n)
        return images[test_fraction:][:5], masks[test_fraction:][:5]
