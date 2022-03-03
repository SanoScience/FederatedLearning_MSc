import pandas as pd

df_split = pd.read_csv('mimic-cxr-2.0.0-split.csv')
df_metadata = pd.read_csv('mimic-cxr-2.0.0-metadata.csv')
df_chexpert = pd.read_csv('mimic-cxr-2.0.0-chexpert.csv')

merge_meta_split = pd.merge(df_split, df_metadata, how='left', left_on=['dicom_id', 'study_id', 'subject_id'],
                            right_on=['dicom_id', 'study_id', 'subject_id'])
merged_full = pd.merge(merge_meta_split, df_chexpert, how='left', left_on=['subject_id', 'study_id'],
                       right_on=['subject_id', 'study_id'])

merged_full['path'] = merged_full.apply(
    lambda r: f'p{str(r["subject_id"])[:2]}/p{r["subject_id"]}/s{r["study_id"]}/{r["dicom_id"]}.jpg', axis=1)

merged_full.to_csv('merged_full.csv', index=False)

print(len(set(merged_full['dicom_id'])))

df_train = merged_full[merged_full['split'] == 'train']
df_test = merged_full[merged_full['split'].isin(['test', 'validate'])]

print(set(merged_full['split']))
print(len(df_train))
print(len(df_test))
print(len(df_train) + len(df_test))

df_train.to_csv('merged_train.csv', index=False)
df_test.to_csv('merged_test.csv', index=False)
