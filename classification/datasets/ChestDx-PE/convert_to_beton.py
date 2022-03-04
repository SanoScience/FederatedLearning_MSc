import os
import sys
import numpy as np

sys.path.append('..')
sys.path.append('../..')

from fl_chestdx_dataset import ChestDxDataset
from utils import get_data_paths, CHESTDX_DATASET_PATH_BASE

from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, NDArrayField

images_dir, test_subset = get_data_paths('chestdx-pe')

dataset = ChestDxDataset(-1, 1, test_subset, images_dir, -1)
write_path = os.path.join(CHESTDX_DATASET_PATH_BASE, 'chestdx-pe-test-jpg90.beton')


writer = DatasetWriter(write_path, {
    'image': RGBImageField(
        write_mode='jpg',
        max_resolution=512,
        jpeg_quality=90,
    ),
    'label': NDArrayField(shape=(15,), dtype=np.dtype('float32'))
}, num_workers=24)
writer.from_indexed_dataset(dataset)
