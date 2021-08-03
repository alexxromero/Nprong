# Based on Yadong's networks. Similar code, I just cleaned up the
# part of the analysis that are no longer needed.

import os
import argparse
import time
import h5py
import numpy as np
from utils import split_data_nsubs

parser = argparse.ArgumentParser(description='Nprong Model')
parser.add_argument(--input_dataf, type=str)
parser.add_argument(--save_dir, type=str)

os.environ['CUDA_VISIBLE_DEVICES'] = torch.cuda.get_device_name(0)
print('training using GPU:', torch.cuda.get_device_name(0))

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from utils import get_batch_classwise_acc

writer = SummaryWriter(args.save_dir)

# -- some tubable variables -- #
device = 'cuda'
lr = 1e-3
epochs = 1000
batch_size = 256


# -- rad data file -- #
with h5py.File(args.input_dataf, 'r') as f:
    y = np.array(f['target'])
    nsubs = np.concatenate((f['Nsubs']['Nsubs_beta05'],
                            f['Nsubs']['Nsubs_beta10'],
                            f['Nsubs']['Nsubs_beta20']), axis=-1)
    mass = np.array(f['jet_Mass'])
    pT = np.array(f['jet_PT'])
    nsamples = y.shape[0]
    nclasses = len(np.unique(y))
    print("Total of {} samples with {} distinct classes".format(nsamples,
                                                                nclasses))

# -- split the data -- #
data_train, data_val, data_test = split_data_nsubs(nsubs, y, mass, pT,
                                                   fold_id=fold_id,
                                                   num_folds=10)
generator = {}
generator['train'] = torch.utils.data(data_train, batch_size=batch_size,
                                      shuffle=True, num_workers=8)
generator['val'] = torch.utils.data(data_val, batch_size=batch_size,
                                    shuffle=True, num_workers=8)
generator['test'] = torch.utils.data(data_test, batch_size=batch_size,
                                     shuffle=True, num_workers=8)

# -- setup the network's architecture -- #
class HLNetBatchNorm(nn.Module):
    def __init__(self, input_dim, inter_dim, num_hidden, out_dim, do_rate):
        super(HLNetBatchNorm, self).__init__
        self.out_dim = out_dim
        modules = []
        for i in range(num_hidden):
            if i == 0:
                modules.append(nn.Linear(input_dim, inter_dim))
            else:
                modules.append(nn.Linear(inter_dim, inter_dim))
            modules.append(nn.ReLu())
            modules.append(nn.BatchNorm1d(inter_dim))
            if do_rate > 0:
                modules.append(nn.Dropout(p=do_rate))
        modules.append(nn.ReLu())
        self.hidden = nn.Sequential(*modules)
        self.output = nn.Linear(inter_dim, out_dim)

    def forward(self, nsubs):
        nsubs = self.hidden(HL)
        return F.relu(self.output(nsubs))

def make_hlnet_base(input_dim, inter_dim, num_hidden, out_dim, do_rate):
    return HLNetBatchNorm(input_dim=input_dim, inter_dim=inter_dim,
                          num_hidden=num_hidden, out_dim=out_dim,
                          do_rate=do_rate)

class HLNet(nn.Module):
    def __init__(self, hlnet_base, out_dim, num_labels=7):
        super(HLNet, self).__init__()
        self.hlnet_base = hlnet_base
        self.top = nn.Linear(out_dim, num_labels)

    def forward(self, nsubs):
        HL = self.hlnet_base(nsubs)
        out = self.top(nsubs)
        return out

hlnet_base = make_hlnet_base(input_dim=nsubs.shape[1],
                             inter_dim=800,
                             num_hidden=5,
                             out_dim=64,
                             do_rate=0.3)

model = HLNet(hlnet_base, out_dim=64, num_labels=nclasses).to(device)


# -- training -- #
hl_param = []
other_param = []
for name, param in model.named_parameters():
    if 'hlnet_base' in name:
        hl_param.append(param)
    else:
        other_param.append(param)

param_groups = [{'params': hl_param, 'lr': lr},
                {'params': other_param, 'lr': lr}]

optimizer = optim.Adam(param_groups, weight_decay=0)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[200, 400, 600, 800], gamma=0.5, last_epoch=-1
    )
loss_fn = nn.CrossEntropyLoss(reduction='none')

def train(model, train_generator, optimizer, epoch):
    model.train()
    for i, (nsubs, y, mass, pT) in enumerate(train_generator):
        optimizer.zero_grad()

        nsubs = torch.tensor(nsubs).float().to(device)
        y = torch.tensor(y).long().to(device)

        ypred = model(nsubs)
        loss = loss_fn(ypred, y)
        loss = loss.mean()
        acc = (torch.argmax(ypred, dim=1) == y).sum().item() / y.shape[0]

        if i%50 == 0:
            print('Train loss: ', loss.item(), '  Train acc: ', acc)
            writer.add_scalar('Loss/train', loss.item(), epoch * iterations + i)
            writer.add_scalar('Acc/train', acc, epoch * iterations + i)

    return loss, acc 
