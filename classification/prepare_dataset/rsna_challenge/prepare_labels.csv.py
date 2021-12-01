import pandas as pd
import random

ids_to_targets_train = dict()
ids_to_targets_all = dict()

train_original_df = pd.read_csv('stage_1_train_labels.csv', delimiter=',')
all_original_df = pd.read_csv('stage_2_train_labels.csv', delimiter=',')

for idx, row in train_original_df.iterrows():
    ids_to_targets_train[row['patientId']] = row['Target']

for idx, row in all_original_df.iterrows():
    ids_to_targets_all[row['patientId']] = row['Target']

test_ids = ids_to_targets_all.keys() - ids_to_targets_train.keys()

train_ids_list = list(ids_to_targets_train)
test_ids_list = list(test_ids)

train_labels = [ids_to_targets_train[idx] for idx in train_ids_list]
test_labels = [ids_to_targets_all[idx] for idx in test_ids_list]


print('All images in train dataset:', len(all_original_df), len(train_labels) + len(test_labels))


train_df = pd.DataFrame(data={'patient_id': train_ids_list, 'label': train_labels})
test_df = pd.DataFrame(data={'patient_id': test_ids_list, 'label': test_labels})
print(train_df.head(10))
print(test_df.head(10))


print('Train dataset:', len(train_ids_list))
print('Test dataset:', len(test_ids_list))

train_df.to_csv('train_labels_stage_1.csv', index=False)
test_df.to_csv('test_labels_stage_1.csv', index=False)

