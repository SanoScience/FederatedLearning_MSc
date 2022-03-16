import os
import sys
import numpy as np

sys.path.append('..')
sys.path.append('../..')

from fl_mimc_dataset import MIMICDataset
from utils import get_data_paths, MIMIC_DATASET_PATH_BASE

from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, NDArrayField

images_dir, train_subset, test_subset = get_data_paths('mimic')

for ds in [('mimic-train-jpg90.beton', train_subset), ('mimic-test-jpg90.beton', test_subset)]:
    dataset = MIMICDataset(-1, 1, ds[1], images_dir, -1)
    write_path = os.path.join(MIMIC_DATASET_PATH_BASE, ds[0])
    writer = DatasetWriter(write_path, {
        'image': RGBImageField(
            write_mode='jpg',
            max_resolution=512,
            jpeg_quality=90,
        ),
        'label': NDArrayField(shape=(13,), dtype=np.dtype('float32'))
    }, num_workers=24)
    writer.from_indexed_dataset(dataset)
