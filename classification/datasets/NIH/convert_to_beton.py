import os
import sys
import numpy as np

sys.path.append('..')
sys.path.append('../..')

from fl_nih_dataset import NIHDataset
from utils import get_data_paths, NIH_DATASET_PATH_BASE

from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, NDArrayField

images_dir, train_subset, test_subset, labels_file = get_data_paths('nih')

for ds in [('nih-train-jpg90.beton', train_subset), ('nih-test-jpg90.beton', test_subset)]:
    dataset = NIHDataset(-1, 1, ds[1], labels_file, images_dir, -1)
    write_path = os.path.join(NIH_DATASET_PATH_BASE, ds[0])

    # Pass a type for each data field
    writer = DatasetWriter(write_path, {
        'image': RGBImageField(
            write_mode='jpg',
            max_resolution=512,
            jpeg_quality=90,
        ),
        'label': NDArrayField(shape=(15,), dtype=np.dtype('float32'))
    }, num_workers=24)

    # Write dataset
    writer.from_indexed_dataset(dataset)
