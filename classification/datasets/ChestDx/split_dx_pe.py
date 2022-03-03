import pandas as pd
import random

full_df = pd.read_csv('chest_dx.csv')
full_df = full_df.rename(columns={'Pleural_thickening': 'Pleural_Thickening', 'No_finding': 'No Finding'})

chest_dx = full_df[full_df['zip'].str.contains('ChestDx-')]
chest_dx_pe = full_df[full_df['zip'].str.contains('ChestDx_PE-')]
chest_dx_pe['patient_id'] = [p[:-8] for p in chest_dx_pe['image']]

# chest_dx_pe.to_csv('../ChestDx-PE/full.csv', index=False)

patients_dx_ids = [p[:-8] for p in chest_dx['image']]
chest_dx['patient_id'] = patients_dx_ids

patients_dx_ids_unique = [p for p in set(patients_dx_ids)]

random.shuffle(patients_dx_ids_unique)

split_idx = int(0.8 * len(patients_dx_ids_unique))

train_patients_ids = set(patients_dx_ids_unique[:split_idx])
test_patients_ids = set(patients_dx_ids_unique[split_idx:])

train_df = chest_dx[chest_dx['patient_id'].isin(train_patients_ids)]
test_df = chest_dx[chest_dx['patient_id'].isin(test_patients_ids)]

# train_df.to_csv('train.csv', index=False)
# test_df.to_csv('test.csv', index=False)
