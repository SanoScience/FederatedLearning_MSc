import pandas as pd

full_train_df = pd.read_csv('train.csv')
full_valid_df = pd.read_csv('valid.csv')

frontal_train_df = full_train_df[full_train_df['Frontal/Lateral'] == 'Frontal']
frontal_valid_df = full_valid_df[full_valid_df['Frontal/Lateral'] == 'Frontal']

frontal_train_df.to_csv('frontal_train.csv', index=False)
frontal_valid_df.to_csv('frontal_valid.csv', index=False)
