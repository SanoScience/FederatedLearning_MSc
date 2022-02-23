import pandas as pd
import os
from shutil import copyfile

RSNA_DATASET_PATH_BASE = os.path.expandvars("$SCRATCH/fl_msc/classification/RSNA/")

images_full_dir = os.path.join(RSNA_DATASET_PATH_BASE, "stage_2_train_images_09_01_png/")
images_segmented_dir = os.path.join(RSNA_DATASET_PATH_BASE, "masked_stage_2_train_images_09_01_1024/")

out_full_dir = '/net/archive/groups/plggsano/fl_msc_classification/rsna/stage_1_train/full'
out_segmented_dir = '/net/archive/groups/plggsano/fl_msc_classification/rsna/stage_1_train/segmented'

df = pd.read_csv('test_labels_stage_1.csv')

for i, r in df.iterrows():
    img = r['patient_id'] + '.png'
    copyfile(os.path.join(images_full_dir, img), os.path.join(out_full_dir, img))
    copyfile(os.path.join(images_segmented_dir, img), os.path.join(out_segmented_dir, img))
