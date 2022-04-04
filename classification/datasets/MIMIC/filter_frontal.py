import pandas as pd

# View positions statistics
# RAO : 3
# XTABLE LATERAL : 2
# AP RLD : 2
# LATERAL : 82853
# PA RLD : 1
# LAO : 3
# LPO : 1
# LL : 35133
# AP : 147173
# SWIMMERS : 1
# AP AXIAL : 2
# PA : 96161
# AP LLD : 2
# PA LLD : 4

full_merged_df = pd.read_csv('merged_full.csv')
print(set(full_merged_df['ViewPosition']))
print()

merged_df = pd.read_csv('merged_full.csv')

for c in ['RAO', 'XTABLE LATERAL', 'AP RLD', 'LATERAL', 'PA RLD', 'LAO', 'LPO', 'LL', 'AP', 'SWIMMERS', 'AP AXIAL',
          'PA', 'AP LLD', 'PA LLD']:
    print(c, ':', len(merged_df[(merged_df['ViewPosition'] == c)]))

print()
print(len(merged_df[(merged_df['ViewPosition'] == 'AP') | (merged_df['ViewPosition'] == 'PA')]))

merged_train_df = pd.read_csv('merged_train.csv')
merged_test_df = pd.read_csv('merged_test.csv')

frontal_train_df = merged_train_df[
    (merged_train_df['ViewPosition'] == 'AP') | (merged_train_df['ViewPosition'] == 'PA')]
frontal_test_df = merged_test_df[(merged_test_df['ViewPosition'] == 'AP') | (merged_test_df['ViewPosition'] == 'PA')]

frontal_train_df.to_csv('frontal_train.csv', index=False)
frontal_test_df.to_csv('frontal_test.csv', index=False)
