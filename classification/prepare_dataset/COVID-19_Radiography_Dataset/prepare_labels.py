# COVID 3616
# Lung_opacity 6012
# Normal 10192
# Viral Pneumonia 1345

import random
import pandas as pd

names_normal = []
names_covid = []

for i in range(1, 3617):
    names_covid.append(f'COVID-{i}.png')

for j in range(1, 10193):
    names_normal.append(f'Normal-{j}.png')


random.shuffle(names_covid)
random.shuffle(names_normal)

train_covid = names_covid[:int(len(names_covid) * 0.8)]
train_normal = names_normal[:int(len(names_normal) * 0.8)]

test_covid = names_covid[int(len(names_covid) * 0.8):]
test_normal = names_normal[int(len(names_normal) * 0.8):]

all_train = train_covid + train_normal
all_test = test_covid + test_normal

random.shuffle(all_train)
random.shuffle(all_test)

labels_train = [1 if 'COVID' in i else 0 for i in all_train]
labels_test = [1 if 'COVID' in i else 0 for i in all_test]

train_df = pd.DataFrame(data={'file_name': all_train, 'label': labels_train})
test_df = pd.DataFrame(data={'file_name': all_test, 'label': labels_test})
print(train_df.head(10))
print(test_df.head(10))


train_df.to_csv('train_labels_covid_normal.csv', index=False)
test_df.to_csv('test_labels_covid_normal.csv', index=False)