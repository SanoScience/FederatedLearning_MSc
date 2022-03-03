import pandas as pd

paths_df = pd.read_csv('cc_cxri_p.csv')
covid_df = pd.read_csv('viral_pneumonia_COVID-19.csv')

covid_dict = {r['image']: r['COVID-19'] for _, r in covid_df.iterrows()}

paths_df['base_class'] = paths_df.apply(lambda r: r['path'].split('-')[0], axis=1)
print(paths_df.head())
paths_df['class'] = paths_df.apply(
    lambda r: 'COVID' if r['base_class'] == 'Viral' and covid_dict[r['path'].split('/')[1]] else r['base_class'],
    axis=1)
# paths_df.to_csv('full.csv', index=False)

df_shuffled = paths_df.sample(frac=1).reset_index(drop=True)
split_idx = int(0.8 * len(df_shuffled))

train_df = df_shuffled[:split_idx]
test_df = df_shuffled[split_idx:]

# train_df.to_csv('train.csv', index=False)
# test_df.to_csv('test.csv', index=False)

# Patients are unique
print(len(paths_df.apply(lambda r: r['path'].split('/')[1], axis=1)))
print(len(set(paths_df.apply(lambda r: r['path'].split('/')[1], axis=1))))
