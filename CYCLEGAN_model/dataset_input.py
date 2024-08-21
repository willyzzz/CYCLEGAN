import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random
import anndata as ad

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TensorDataset(Dataset):
    def __init__(self, tensor1, tensor2):
        assert tensor1.size(0) == tensor2.size(0)
        self.tensor1 = tensor1
        self.tensor2 = tensor2

    def __len__(self):
        return self.tensor1.size(0)  # Number of samples

    def __getitem__(self, idx):
        return self.tensor1[idx], self.tensor2[idx]

def replace_zeros_with_row_mean(row):
    # 计算当前行非零值的均值
    mean_value = row[row != 0].mean()
    # 将 0 值替换为均值
    return row.apply(lambda x: mean_value if x == 0 else x)

def generate_stratified_proportions(df, n_samples_per_set=1000, n_sets=100, noise_level=2):
    proportions = []

    for _ in range(n_sets):
        stratified_sample = df.groupby('0', group_keys=False).apply(
            lambda x: x.sample(frac=n_samples_per_set / len(df), replace=True))
        prop = stratified_sample['0'].value_counts(normalize=True)

        # 添加随机扰动
        prop_dict = prop.to_dict()
        for cell_type in prop_dict:
            prop_dict[cell_type] += np.random.uniform(-noise_level, noise_level)

        # 确保比例在0到1之间并归一化
        total = sum(prop_dict.values())
        for cell_type in prop_dict:
            prop_dict[cell_type] = max(0, min(prop_dict[cell_type], 1)) / total

        # 再次归一化，确保总和为1
        total = sum(prop_dict.values())
        for cell_type in prop_dict:
            prop_dict[cell_type] /= total

        proportions.append(prop_dict)

    proportion_df = pd.DataFrame(proportions).fillna(0)

    return proportion_df

def fra_ini_generate(counts_path_list, num_samples_bulk):
    counts_list = []
    for count_path in counts_path_list:
        count = pd.read_csv(count_path, sep='\t', index_col=0)
        counts_list.append(count)
    num_repeats = num_samples_bulk // len(counts_path_list)
    num_extra = num_samples_bulk % len(counts_path_list)
    results = []
    for count in counts_list:
        # sampling = count.sample(n=1000)
        # data = sampling['0'].value_counts()
        # cell_counts = pd.Series(data, name="Celltype")
        # cell_proportions = cell_counts / cell_counts.sum()
        # df_proportions = pd.DataFrame(cell_proportions).T
        # df_proportions = df_proportions[sorted(df_proportions.columns)]
        # results.append(df_proportions)
        proportion_data = generate_stratified_proportions(count, n_samples_per_set=1000, n_sets=num_repeats, noise_level=0.05)
        results.append(proportion_data)


    if num_extra == 0:
        pass
    else:
        proportion_data = generate_stratified_proportions(count, n_samples_per_set=1000, n_sets=num_extra,
                                                          noise_level=0.05)
        results.append(proportion_data)

    fra_ini = pd.concat(results, ignore_index=True)
    fra_ini = fra_ini.fillna(0)
    fra_ini = fra_ini.sort_index(axis=1)
    fra_ini = fra_ini.values

    return fra_ini

def pbmc_dataset_process(data_path_all, mode='train', log_icon=False, minmax_icon=True, fra_ini_icon='sampling', batchsize=32):

    ###1. data_path import
    sig_path = data_path_all[0]
    train_bulk_path = data_path_all[1]
    test_bulk_path = data_path_all[2]
    train_fra_gt_path = data_path_all[3]
    test_fra_gt_path = data_path_all[4]
    fra_ini_path = data_path_all[5]
    counts_path_list = data_path_all[6]

    ###2.data import
    sig = pd.read_csv(sig_path, sep = ',', index_col=0)
    if train_bulk_path.endswith('txt'):
        train_bulk = pd.read_csv(train_bulk_path, sep = '\t', index_col=0)
    elif train_bulk_path.endswith('h5ad'):
        adata = ad.read_h5ad(train_bulk_path)
        train_bulk = pd.DataFrame(adata.X, columns = adata.var_names)
    train_fra_gt = pd.read_csv(train_fra_gt_path, sep = '\t', index_col=0)
    test_bulk = pd.read_csv(test_bulk_path, sep = '\t', index_col=0)
    test_fra_gt = pd.read_csv(test_fra_gt_path, sep = '\t', index_col=0)
    # sig = sig.apply(replace_zeros_with_row_mean, axis=1)
    if train_bulk.shape[0] > 10000:
        train_bulk = train_bulk.head(10000)
        train_fra_gt = train_fra_gt.head(10000)

    ###3.common genes filter
    common_columns = (sig.columns.intersection(train_bulk.columns)).intersection(test_bulk.columns)
    sig = sig[common_columns].values
    train_bulk = train_bulk[common_columns].values
    train_fra_gt = train_fra_gt.values
    test_bulk = test_bulk[common_columns].values
    test_fra_gt = test_fra_gt.values
    print('***%d genes intersection' % len(common_columns))

    ###4.train/test mode decide
    if mode == 'train':
        parts = train_bulk_path.split('/')
        dataset_name = parts[-1]
        print('***mode is training, training dataset is: ', dataset_name)
        bulk = train_bulk
        fra_gt = train_fra_gt
    else:
        parts = test_bulk_path.split('/')
        dataset_name = parts[-1]
        print('***mode is testing, testing dataset is: ', dataset_name)
        bulk = test_bulk
        fra_gt = test_fra_gt

    num_samples_bulk = bulk.shape[0]
    num_cell_type = sig.shape[0]

    ###5.log2 transform decide
    if log_icon == True:
        bulk = np.log2(bulk+1)
        print('***bulk log_transform done')
        if 'log' not in sig_path:
            sig = np.log2(sig + 1)
            print('***sig log transform done')
    # else:
    #     bulk = bulk + 0.01
    #     sig = sig + 0.01

    ###6.minmax transform decide
    if minmax_icon == True:
        mms = MinMaxScaler()
        bulk = mms.fit_transform(bulk.T)
        bulk = bulk.T
        sig = mms.fit_transform(sig.T)
        sig = sig.T
        print('max bulk is: ', np.max(bulk))
        print('max sig is: ', np.max(sig))
        print('***minmax transform of bulk and sig done')

    ###7.fraction initial input decide
    if mode == 'train':
        fra_ini = fra_gt
        print('***real fractions load')

    elif mode == 'test':
        if fra_ini_icon == 'sampling':
            fra_ini = fra_ini_generate(counts_path_list, num_samples_bulk)
        elif fra_ini_icon == 'rand':
            fra_ini = np.random.rand(*test_fra_gt.shape)
            fra_ini = fra_ini / fra_ini.sum(axis=1, keepdims=True)
        else:
            fra_ini = pd.read_csv(fra_ini_path, sep=',', index_col=0)
            num_repeats = num_samples_bulk // fra_ini.shape[0]
            fra_ini = pd.concat([fra_ini] * num_repeats, ignore_index=True)
            fra_ini = fra_ini.values
        print('fra_ini: ', fra_ini)
        print('***real fractions augmentation done')

    bulk = torch.tensor(bulk, dtype=torch.float32).to(device)
    sig = torch.tensor(sig, dtype=torch.float32).to(device)
    fra_gt = torch.tensor(fra_gt, dtype=torch.float32).to(device)
    fra_ini = torch.tensor(fra_ini, dtype=torch.float32).to(device)

    dataset = TensorDataset(bulk, fra_ini)
    print('***dataset load success')
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)

    flattened_samples = sig.flatten().cpu().numpy()
    plt.figure(figsize=(10, 6))
    plt.hist(flattened_samples, bins=100, density=True, alpha=0.6, color='g')
    plt.title('Histogram of shifted Gaussian distribution samples')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    flattened_samples = bulk.flatten().cpu().numpy()
    plt.hist(flattened_samples, bins=100, density=True)
    plt.title("simulated-bulk")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

    return dataloader, bulk, sig, fra_gt, fra_ini




def cancer_dataset_process(data_path_all, log_icon=False, minmax_icon=True, fra_ini_icon='sampling', add_noise = True, batchsize=32):

    ###1. data_path import
    sig_path = data_path_all[0]
    test_bulk_path = data_path_all[1]
    test_fra_gt_path = data_path_all[2]
    ini_fra_path_list = data_path_all[3]

    ###2.data import
    sig = pd.read_csv(sig_path, sep = '\t', index_col=0)
    test_bulk = pd.read_csv(test_bulk_path, sep = '\t', index_col=0)
    test_fra_gt = pd.read_csv(test_fra_gt_path, sep = '\t', index_col=0)
    test_bulk = test_bulk.loc[:, ~test_bulk.columns.duplicated()]

    sig = sig.apply(replace_zeros_with_row_mean, axis=1)

    ###3.common genes filter
    common_columns = sig.columns.intersection(test_bulk.columns)
    sig = sig[common_columns].values
    bulk = test_bulk[common_columns].values
    fra_gt = test_fra_gt.values
    print('***%d genes intersection' % len(common_columns))

    if add_noise == True:
        mean_noise = 0
        std_noise = 0.01  # 设定标准差，你可以根据需要调整
        noise = np.random.normal(mean_noise, std_noise, sig.shape)
        sig = sig + 0.01*noise

    num_samples_bulk = bulk.shape[0]
    num_cell_type = sig.shape[0]

    ###5.log2 transform decide
    if log_icon == True:
        bulk = np.log2(bulk+1)
        print('***bulk log_transform done')
        if 'log' not in sig_path:
            sig = np.log2(sig + 1)
            print('***sig log transform done')
    # else:
    #     bulk = bulk + 0.01
    #     sig = sig + 0.01

    ###6.minmax transform decide
    if minmax_icon == True:
        mms = MinMaxScaler()
        bulk = mms.fit_transform(bulk.T)
        bulk = bulk.T
        sig = mms.fit_transform(sig.T)
        sig = sig.T
        print('max bulk is: ', np.max(bulk))
        print('max sig is: ', np.max(sig))
        print('***minmax transform of bulk and sig done')

    ###7.fraction initial input decide
    if fra_ini_icon == 'sampling':
        fra_ini = fra_ini_generate(ini_fra_path_list, num_samples_bulk)
    elif fra_ini_icon == 'rand':
        fra_ini = np.random.rand(test_fra_gt.shape[0], 3)
        fra_ini = fra_ini / fra_ini.sum(axis=1, keepdims=True)

    print('fra_ini: ', fra_ini)
    print('***real fractions augmentation done')

    bulk = torch.tensor(bulk, dtype=torch.float32).to(device)
    sig = torch.tensor(sig, dtype=torch.float32).to(device)
    fra_gt = torch.tensor(fra_gt, dtype=torch.float32).to(device)
    fra_ini = torch.tensor(fra_ini, dtype=torch.float32).to(device)

    dataset = TensorDataset(bulk, fra_ini)
    print('***dataset load success')
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=False)

    flattened_samples = sig.flatten().cpu().numpy()
    plt.figure(figsize=(10, 6))
    plt.hist(flattened_samples, bins=100, density=True, alpha=0.6, color='g')
    plt.title('Histogram of shifted Gaussian distribution samples')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    flattened_samples = bulk.flatten().cpu().numpy()
    plt.hist(flattened_samples, bins=100, density=True)
    plt.title("simulated-bulk")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

    return dataloader, bulk, sig, fra_gt, fra_ini


