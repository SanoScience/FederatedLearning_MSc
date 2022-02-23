import pandas as pd

train_df = pd.read_csv('train_labels_all_classes.csv')
test_df = pd.read_csv('test_labels_all_classes.csv')

id_to_class_name = {0: "Normal",
                    1: "COVID",
                    2: "Lung_Opacity",
                    3: "Viral Pneumonia"}

split_labels = []
image_paths = []
labels = []
for i, r in train_df.iterrows():
    split_labels.append('TRAIN')
    image_paths.append('gs://automl-covid-dataset/COVID_19_all_classes/'+r['file_name'])
    labels.append(id_to_class_name[r['label']])

print(len(train_df))

for i, r in test_df.iterrows():
    if i % 2 == 0:
        split_labels.append('TEST')
    else:
        split_labels.append('VALIDATION')
    image_paths.append('gs://automl-covid-dataset/COVID_19_all_classes/'+r['file_name'])
    labels.append(id_to_class_name[r['label']])

df = pd.DataFrame(data={'set': split_labels, 'image_path': image_paths, 'label': labels})

df.to_csv('automl_dataset.csv', index=False)
