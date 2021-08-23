# Based on Yadong's networks. Similar code, I just cleaned up the
# part of the analysis that are no longer needed.

import os
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MaxAbsScaler


def get_nsub_mass_dataset(input_file):
    mass_pT_file = "/home/alex/Desktop/Nprong_AR/datasets/dataset_noPtNorm.h5"
    with h5py.File(mass_pT_file, 'r') as f:
        mass = np.array(f['jet_Mass'])
        pT = np.array(f['jet_PT'])
    with h5py.File(input_file, 'r') as f:
        y = np.array(f['target'])
        nsubs_mass = np.concatenate((f['Nsubs']['Nsubs_beta05'],
                                     f['Nsubs']['Nsubs_beta10'],
                                     f['Nsubs']['Nsubs_beta20'],
                                     mass.reshape(-1, 1)), axis=-1)
    return y, nsubs_mass, mass, pT


def get_nsub_dataset(input_file):
    mass_pT_file = "/home/alex/Desktop/Nprong_AR/datasets/dataset_noPtNorm.h5"
    with h5py.File(mass_pT_file, 'r') as f:
        mass = np.array(f['jet_Mass'])
        pT = np.array(f['jet_PT'])
    with h5py.File(input_file, 'r') as f:
        y = np.array(f['target'])
        nsubs = np.concatenate((f['Nsubs']['Nsubs_beta05'],
                                f['Nsubs']['Nsubs_beta10'],
                                f['Nsubs']['Nsubs_beta20']), axis=-1)
        #mass = np.array(f['jet_Mass'])
        #pT = np.array(f['jet_PT'])
    return y, nsubs, mass, pT

def get_threeM_dataset(input_file):
    mass_pT_file = "/home/alex/Desktop/Nprong_AR/datasets/dataset_noPtNorm.h5"
    with h5py.File(mass_pT_file, 'r') as f:
        mass = np.array(f['jet_Mass'])
        pT = np.array(f['jet_PT'])
    with h5py.File(input_file, 'r') as f:
        y = np.array(f['target'])
        threeM = np.array(f['Constituents']['threeM'])
    return y, threeM, mass, pT

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


class Dataset(Dataset):
    def __init__(self, X, y, mass, pT):
        self.X, self.y, self.mass, self.pT = X, y, mass, pT
        self.n = X.shape[0]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index], self.mass[index], self.pT[index]


def split_dataset(X, y, mass, pT, fold_id=None, num_folds=10, scale=False):
    total_num_sample = X.shape[0]
    if fold_id is not None:
        l = [X, y, mass, pT]
        # Rearrange the samples. The test samples are placed at the end.
        for i in range(len(l)):
            l[i] = cross_validate_perm(l[i], fold_id=fold_id,
                                       num_folds=num_folds)
        X, y, mass, pT = l

    # for 10-folds, the split train:test:val is 80:10:10
    train_cut = int(total_num_sample * 0.8)
    val_cut = int(total_num_sample * 0.9)

    X_train, X_val, X_test = \
        X[:train_cut], X[train_cut:val_cut], X[val_cut:]
    y_train, y_val, y_test = \
        y[:train_cut], y[train_cut:val_cut], y[val_cut:]
    mass_train, mass_val, mass_test = \
        mass[:train_cut], mass[train_cut:val_cut], mass[val_cut:]
    pT_train, pT_val, pT_test = \
        pT[:train_cut], pT[train_cut:val_cut], pT[val_cut:]

    if scale:
        scaler = StandardScaler()
        #scaler = MaxAbsScaler()
        nsubs_train = scaler.fit_transform(X_train)
        nsubs_val = scaler.transform(X_val)
        nsubs_test = scaler.transform(X_test)

    data_train = Dataset(X_train, y_train, mass_train, pT_train)
    data_val = Dataset(X_val, y_val, mass_val, pT_val)
    data_test = Dataset(X_test, y_test, mass_test, pT_test)

    return data_train, data_val, data_test


def plot_loss_acc(train_loss, train_acc, val_loss, val_acc,
                  save_dir, model_tag):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ix = np.argmax(val_acc)
    ax[0].plot(train_acc, label='Train')
    ax[0].plot(val_acc, label='Val')
    ax[0].vlines(ix, 
                 np.min([np.min(train_acc), np.min(val_acc)]), 
                 np.max([np.max(train_acc), np.max(val_acc)]),
                 colors='black', linestyles='dotted')
    ax[0].set_xlabel('epoch', fontsize=14)
    ax[0].set_ylabel('Accuracy', fontsize=14)
    ax[0].legend(fontsize=14)

    ax[1].plot(train_loss, label='Train')
    ax[1].plot(val_loss, label='Val')
    ax[1].vlines(ix,
                 np.min([np.min(train_loss), np.min(val_loss)]),
                 np.max([np.max(train_loss), np.max(val_loss)]),
                 colors='black', linestyles='dotted')
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
