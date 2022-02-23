import json
import pandas as pd
import os
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split

path_to_jsons = "/Users/przemyslawjablecki/Downloads/all-images"
classes = ['COVID-19', 'No Finding', 'Undefined Pneumonia', 'No Pneumonia (healthy)', 'Varicella',
           'Pneumonia', 'SARS']


def in_classes(item):
    for c in classes:
        if c.lower() in item.lower():
            return True
    return False


files = os.listdir(path_to_jsons)
labels = set()
c = Counter()
label_dict = defaultdict(list)

for f in files:
    with open(os.path.join(path_to_jsons, f)) as file:
        d = json.load(file)
        annotations = d['annotations']
        names = list(map(lambda item: item['name'], annotations))
        annotations = list(filter(in_classes, names))
        filename = d['image']['filename']
        if len(annotations) == 0:
            # print(d['image']['filename'], print('dataset'), f, list(names))
            # c['no_class'] += 1
            continue

        if len(annotations) == 2 and ('Viral Pneumonia' in annotations and 'COVID-19' in annotations):
            c['COVID-19'] += 1
            label_dict['filename'].append(filename)
            label_dict['class'].append('COVID-19')
            continue
        if len(annotations) == 2 and ('Pneumonia' in annotations and 'Undefined Pneumonia' in annotations):
            c['Pneumonia'] += 1
            label_dict['filename'].append(filename)
            label_dict['class'].append('Pneumonia')
            continue
        if len(annotations) == 2 and ('Viral Pneumonia' in annotations and 'SARS' in annotations):
            c['SARS'] += 1
            label_dict['filename'].append(filename)
            label_dict['class'].append('SARS')
            continue
        if len(annotations) == 2 and ('Viral Pneumonia' in annotations and 'Varicella' in annotations):
            c['Varicella'] += 1
            label_dict['filename'].append(filename)
            label_dict['class'].append('Varicella')
            continue
        if len(annotations) == 2 and 'No Pneumonia (healthy)' in annotations and 'No Finding' in annotations:
            c['No Pneumonia (healthy)'] += 1
            label_dict['filename'].append(filename)
            label_dict['class'].append('No Pneumonia (healthy)')
            continue

        if len(annotations) > 1:
            c['multi'] += 1
            print(d['image']['filename'], print('dataset'), f, annotations)
            continue
        for a in annotations:
            c[a] += 1
            label_dict['filename'].append(filename)
            label_dict['class'].append(a)

print(c)

X = label_dict['filename']
y = label_dict['class']
x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.1,
                                                    random_state=42,
                                                    stratify=y)
split_dict = defaultdict(list)
for f, cls in zip(x_train, y_train):
    split_dict['filename'].append(f)
    split_dict['class'].append(cls)
    split_dict['train'].append(True)
for f, cls in zip(x_test, y_test):
    split_dict['filename'].append(f)
    split_dict['class'].append(cls)
    split_dict['train'].append(False)

df = pd.DataFrame.from_dict(split_dict)

print(df.head())
df.to_csv('labels.csv')
df = pd.read_csv('labels.csv')

for index, row in df.iterrows():
    print(row['class'], row['filename'], row['train'])
    print(type(row['train']))
    print(row['train'] == True)
    print(row['train'] == False)
    break

