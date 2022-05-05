import pandas as pd
from collections import Counter

labels = '/Users/przemyslawjablecki/PycharmProjects/FederatedLearning_MSc/segmentation/labels.csv'
df = pd.read_csv(labels)
print(df.head())
counter = Counter(df['class'])
print(counter)
print(len(df['class']))
print(counter.keys())