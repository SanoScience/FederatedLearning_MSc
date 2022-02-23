# Normal 10192
# COVID 3616
# Lung_opacity 6012
# Viral Pneumonia 1345

import random
import pandas as pd

names_normal = []
names_covid = []
names_viral_pneumonia = []
names_lung_opacity = []

class_name_to_id = {"Normal": 0,
                    "COVID": 1,
                    "Lung_Opacity": 2,
                    "Viral Pneumonia": 3}

class_filenames = {i: [] for i in range(4)}

for i in range(1, 10193):
    class_filenames[0].append(f'Normal-{i}.png')

for i in range(1, 3617):
    class_filenames[1].append(f'COVID-{i}.png')

for i in range(1, 6013):
    class_filenames[2].append(f'Lung_Opacity-{i}.png')

for i in range(1, 1346):
    class_filenames[3].append(f'Viral Pneumonia-{i}.png')

train_filenames = {}
test_filenames = {}

for i in range(4):
    random.shuffle(class_filenames[i])
    train_filenames[i] = class_filenames[i][:int(len(class_filenames[i]) * 0.8)]
    test_filenames[i] = class_filenames[i][int(len(class_filenames[i]) * 0.8):]

all_train = []
all_test = []

for i in range(4):
    all_train.extend(train_filenames[i])
    all_test.extend(test_filenames[i])

random.shuffle(all_train)
random.shuffle(all_test)

labels_train = [class_name_to_id[k.split('-')[0]] for k in all_train]
labels_test = [class_name_to_id[k.split('-')[0]] for k in all_test]

train_df = pd.DataFrame(data={'file_name': all_train, 'label': labels_train})
test_df = pd.DataFrame(data={'file_name': all_test, 'label': labels_test})
print(train_df.head(10))
print(test_df.head(10))

train_df.to_csv('train_labels_all_classes.csv', index=False)
test_df.to_csv('test_labels_all_classes.csv', index=False)
