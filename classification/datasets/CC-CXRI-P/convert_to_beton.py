import os
import sys

sys.path.append('..')
sys.path.append('../..')

from fl_cc_cxri_p_dataset import CCCXRIPDataset
from utils import get_data_paths, CHESTDX_DATASET_PATH_BASE

from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField

images_dir, test_subset = get_data_paths('cc-cxri-p')

dataset = CCCXRIPDataset(-1, 1, test_subset, images_dir, -1)
write_path = os.path.join(CHESTDX_DATASET_PATH_BASE, 'cc-cxri-p-test-jpg90.beton')

writer = DatasetWriter(write_path, {
    'image': RGBImageField(
        write_mode='jpg',
        max_resolution=512,
        jpeg_quality=90,
    ),
    'label': IntField()
}, num_workers=24)

# Write dataset
writer.from_indexed_dataset(dataset)
