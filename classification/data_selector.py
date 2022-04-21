from abc import ABC, abstractmethod


class DataSelector(ABC):
    def __init__(self):
        pass

    def select_data(self, images, masks, client_id, number_of_clients):
        pass


class IIDSelector(DataSelector):
    def select_data(self, images, labels, client_id, number_of_clients):
        sampled_images = [path for i, path in enumerate(images) if i % number_of_clients == client_id]
        sampled_labels = [label for i, label in enumerate(labels) if i % number_of_clients == client_id]
        return sampled_images, sampled_labels

    def get_ids(self, dataset_len, client_id, number_of_clients):
        return [i for i in range(dataset_len) if i % number_of_clients == client_id]


class IncreasingSelector(DataSelector):
    def get_ids(self, dataset_len, client_id, number_of_clients):
        chunks_count = number_of_clients * (number_of_clients + 1) // 2
        chunk_size = dataset_len // chunks_count
        start_id = chunk_size * client_id * (client_id + 1) // 2
        end_id = start_id + chunk_size * (client_id + 1)
        return [i for i in
                (range(start_id, end_id) if client_id < (number_of_clients - 1) else range(start_id, dataset_len))]
