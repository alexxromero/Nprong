# Based on Yadong's networks. Similar code, I just cleaned up the
# part of the analysis that are no longer needed.

import os
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt

def get_nsub_dataset(input_file):
    with h5py.File(input_file, 'r') as f:
        y = np.array(f['target'])
        nsubs = np.concatenate((f['Nsubs']['Nsubs_beta05'],
                                f['Nsubs']['Nsubs_beta10'],
                                f['Nsubs']['Nsubs_beta20']), axis=-1)
        mass = np.array(f['jet_Mass'])
        pT = np.array(f['jet_PT'])
    return y, nsubs, mass, pT


def get_acc_per_class(ypred, y):
    class_accs = []
    for i in range(1, len(np.unique(y))):
        inbin = np.where(y==i)[0]
        ypred_bin = ypred[inbin]
        y_bin = y[inbin]
        acc = np.where(ypred_bin == y_bin)[0].shape[0]
        class_accs.append(acc / inbin.shape[0])
    return class_accs

def get_acc_per_massbin(ypred, y, mass):
    bins = [300+i*50 for i in range(9)]
    mass_ix = np.digitize(mass, bins)
    massbin_accs = []
    for i in range(1, len(bins)):
        inbin = np.where(mass_ix==i)[0]
        ypred_bin = ypred[inbin]
        y_bin = y[inbin]
        acc = np.where(ypred_bin == y_bin)[0].shape[0]
        massbin_accs.append(acc / inbin.shape[0])
    return massbin_accs

def get_acc_per_pTbin(ypred, y, pT):
    bins = [1000+i*20 for i in range(11)]
    pT_ix = np.digitize(pT, bins)
    pTbin_accs = []
    for i in range(1, len(bins)):
        inbin = np.where(pT_ix==i)[0]
        ypred_bin = ypred[inbin]
        y_bin = y[inbin]
        acc = np.where(ypred_bin == y_bin)[0].shape[0]
        pTbin_accs.append(acc / inbin.shape[0])
    return pTbin_accs


class NsubsDataset(Dataset):
    def __init__(self, nsubs, y, mass, pT):
        self.nsubs, self.y, self.mass, self.pT = nsubs, y, mass, pT
        self.n = nsubs.shape[0]

    def __len__(self):
        return len(self.nsubs)

    def __getitem__(self, index):
        return self.nsubs[index], self.y[index], self.mass[index], self.pT[index]


def split_nsub_dataset(nsubs, y, mass, pT, fold_id=None, num_folds=10):
    total_num_sample = nsubs.shape[0]
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

    data_train = NsubsDataset(nsubs_train, y_train, mass_train, pT_train)
    data_val = NsubsDataset(nsubs_val, y_val, mass_val, pT_val)
    data_test = NsubsDataset(nsubs_test, y_test, mass_test, pT_test)

    return data_train, data_val, data_test


def plot_loss_acc(train_loss, train_acc, val_loss, val_acc,
                  save_dir, model_tag):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(train_acc, label='Train')
    ax[0].plot(val_acc, label='Val')
    ax[0].set_xlabel('epoch', fontsize=14)
    ax[0].set_ylabel('Accuracy', fontsize=14)
    ax[0].legend(fontsize=14)

    ax[1].plot(train_loss, label='Train')
    ax[1].plot(val_loss, label='Val')
    ax[1].set_xlabel('epoch', fontsize=14)
    ax[1].set_ylabel('Loss', fontsize=14)
    ax[1].legend(fontsize=14)

    plt.savefig(os.path.join(save_dir, "acc_loss_{}.png".format(model_tag)),
                bbox_inches='tight')


def cross_validate_perm(data, fold_id, num_folds=10):
    n = data.shape[0]
    test_size = int(n // num_folds)
    test_ids = (fold_id * test_size + np.arange(test_size)).tolist()
    perm = np.array([i for i in range(n) if i not in test_ids] + test_ids)
    return data[perm]
