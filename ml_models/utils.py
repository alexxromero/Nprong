# Based on Yadong's networks. Similar code, I just cleaned up the
# part of the analysis that are no longer needed.

import numpy as np
from torch.utils.data import Dataset
import torch


def cross_validate_perm(data, fold_id, num_folds=10):
    n = data.shape[0]
    tests_size = int(n // num_folds)
    test_ids = (fold_id * test_size + np.arange(test_size)).tolist()
    perm = np.array([i for i in range(n) if i not in test_ids] + test_ids)
    return data[perm]


def split_data_nsubs(nsubs, y, mass, pT, fold_id=None, num_folds=10):
    n = nsubs.shape[0]
    if fold_id is not None:
        l = [nsubs, y, mass, pT]
        # Rearrange the samples. The test samples are placed at the end.
        for i in range(len(l)):
            l[i] = cross_validate_perm(l[i], fold_id=fold_id,
                                       num_folds=num_folds)
        nsubs, y, mass, pT = l

    # for 10-folds, the split train:test:val is 80:10:10
    train_cut = int(total_num_sample * 0.8)
    val_cut = int(total_num_sample * 0.9)

    nsubs_train, nsubs_val, nsubs_test = \
        nsubs[:train_cut], nsubs[train_cut:val_cut], nsubs[val_cut:]
    y_train, y_val, y_test = \
        y[:train_cut], y[train_cut:val_cut], y[val_cut:]
    mass_train, mass_val, mass_test = \
        mass[:train_cut], mass[train_cut:val_cut], mass[val_cut:]
    pT_train, pT_val, pT_test = \
        pT[:train_cut], pT[train_cut:val_cut], pT[val_cut:]

    data_train = EFPDataset(nsubs_train, y_train, mass_train, pT_train)
    data_val = EFPDataset(nsubs_val, y_val, mass_val, pT_val)
    data_test = EFPDataset(nsubs_test, y_test, mass_test, pT_test)

    return data_train, data_val, data_test


class NsubsDataet(Dataset):
    def __init__(self, nsubs, y, mass, pT):
        self.nsubs, self.y, self.mass, self.pT = nsubs, y, mass, pT
        self.n = nsubs.shape[0]

    def __len__(self):
        return len(self.nsubs)

    def __getitem__(self, index):
        return self.nsubs[index], self.y[index], self.mass[index], self.pT[index]
