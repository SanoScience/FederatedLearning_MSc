import os
import sys

sys.path.append('..')
sys.path.append('../..')

from fl_rsna_dataset import RSNADataset
from utils import get_data_paths, RSNA_DATASET_PATH_BASE

from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField

images_dir, train_subset, test_subset = get_data_paths('rsna-full')

for ds in [('train.beton', train_subset), ('test.beton', test_subset)]:
    dataset = RSNADataset(-1, 1, ds[1], images_dir, None, -1)
    write_path = os.path.join(RSNA_DATASET_PATH_BASE, ds[0])

    # Pass a type for each data field
    writer = DatasetWriter(write_path, {
        'image': RGBImageField(
            write_mode='raw',
            max_resolution=512,
        ),
        'label': IntField()
    }, num_workers=8)

    # Write dataset
    writer.from_indexed_dataset(dataset)
