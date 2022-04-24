import re
import pandas as pd
import numpy as np
import os

pattern = r'r_(\d*)-c_(\d*)_bs_(\d*)_le_(\d*)_fs_([^_]*)_mf_(\d*)_ff_(\d\.\d*)_do_False_o_([^_]*)_lr_(\d\.\d*)'
results_dir = ''

for filename in os.listdir(results_dir):
    re_found = re.search(pattern, filename)
    groups = re_found.groups()
    rounds = groups[0]
    clients = groups[1]
    bs = groups[2]
    le = groups[3]
    aggregation = groups[4]
    mf = groups[5]
    ff = groups[6]
    optimizer = groups[7]
    lr = groups[8]
    result_dir = os.path.join(results_dir, filename)
    df = pd.read_csv(f'{result_dir}/result.csv', index_col=0)
