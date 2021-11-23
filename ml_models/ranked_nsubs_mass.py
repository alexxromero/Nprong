# Based on Yadong's networks. Similar code, I just cleaned up the
# part of the analysis that are no longer needed.

import os
import sys
import argparse
import time
import h5py
import numpy as np
from utils import get_nsub_mass_dataset, split_dataset
from utils import plot_loss_acc
from utils import get_acc_per_class, get_acc_per_massbin, get_acc_per_pTbin
from utils import Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
#from utils import get_batch_classwise_acc

os.environ['CUDA_VISIBLE_DEVICES'] = torch.cuda.get_device_name(0)
print('training using GPU:', torch.cuda.get_device_name(0))

def load_generator(data_train, data_val, data_test, batch_size=256):
    generator = {}
    generator['train'] = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=True, num_workers=8)
    generator['val'] = torch.utils.data.DataLoader(
        data_val, batch_size=batch_size, shuffle=True, num_workers=8)
    generator['test'] = torch.utils.data.DataLoader(
        data_test, batch_size=batch_size, shuffle=True, num_workers=8)
    return generator


class HLNetBatchNorm(nn.Module):
    def __init__(self, input_dim, inter_dim, num_hidden, out_dim, do_rate):
        super(HLNetBatchNorm, self).__init__()
        self.out_dim = out_dim
        modules = []
        for i in range(num_hidden):
            if i == 0:
                modules.append(nn.Linear(input_dim, inter_dim))
            else:
                modules.append(nn.Linear(inter_dim, inter_dim))
            modules.append(nn.ReLU())
            modules.append(nn.BatchNorm1d(inter_dim))
            if do_rate > 0:
                modules.append(nn.Dropout(p=do_rate))
        modules.append(nn.ReLU())
        self.hidden = nn.Sequential(*modules)
        self.output = nn.Linear(inter_dim, out_dim)

    def forward(self, nsubs):
        nsubs = self.hidden(nsubs)
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
        nsubs = self.hlnet_base(nsubs)
        out = self.top(nsubs)
        return out


def test(model, generator, loss_fn, device, epoch=None, validate=False):
    model.eval()
    predy_truey_mass_pT = []
    test_loss, test_acc = 0.0, 0.0
    nsamples = 0
    with torch.no_grad():
        for i, (nsubs, y, mass, pT) in enumerate(generator):
            nsubs = torch.tensor(nsubs).float().to(device)
            y = torch.tensor(y).long().to(device)
            ypred = model(nsubs)
            loss = loss_fn(ypred, y)
            pred_y = torch.argmax(ypred, dim=1)
            predy_truey_mass_pT.append(torch.stack(
                [pred_y.float().cpu(), y.float().cpu(), mass.float(), pT.float()])
                )
            test_loss += loss.sum().item()
            test_acc  += (torch.argmax(ypred, dim=1) == y).sum().item()
            nsamples += y.shape[0]
    predy_truey_mass_pT = torch.cat(predy_truey_mass_pT, dim=1).numpy()
    test_loss /= nsamples  # avg over all samples
    test_acc  /= nsamples  # avg over all samples
    if validate:
        writer.add_scalar('Loss/val', test_loss, epoch)
        writer.add_scalar('Acc/val', test_acc, epoch)
    return test_loss, test_acc, predy_truey_mass_pT


def main(model, hl_save_dir, hl_model_tag, epochs):
    hl_param = []
    other_param = []
    for name, param in model.named_parameters():
        if 'hlnet_base' in name:
            hl_param.append(param)
        else:
            other_param.append(param)
    lr = 1e-4
    param_groups = [{'params': hl_param, 'lr': lr},
                    {'params': other_param, 'lr': lr}]

    optimizer = optim.Adam(param_groups, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[200, 400, 600, 800], gamma=0.5, last_epoch=-1
        )
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    #hl_save_dir = "nsubs_mass_fullrun"
    #hl_model_tag = "nsubs_mass"
    saved_best_model = os.path.join(hl_save_dir,
                                    "best_model_{}.pt".format(hl_model_tag))

    print('Loading best model...')
    cp = torch.load(saved_best_model)
    model.load_state_dict(cp, strict=True)

    _, test_acc, predy_truey_mass_pT = test(
        model, generator['test'], loss_fn, device
        )
    print("Total Acc: ", test_acc)

    predy = predy_truey_mass_pT[0, :]
    truey = predy_truey_mass_pT[1, :]
    mass = predy_truey_mass_pT[2, :]
    pT = predy_truey_mass_pT[3, :]

    class_acc = get_acc_per_class(predy, truey)
    print("Class Acc: ", class_acc)

    mass_acc = get_acc_per_massbin(predy, truey, mass)
    print("Mass-bin Acc: ", mass_acc)

    pT_acc = get_acc_per_pTbin(predy, truey, pT)
    print("pT-bin Acc: ", pT_acc)

    return test_acc, class_acc, mass_acc, pT_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nprong Model')
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--tag", type=str)
    parser.add_argument("--k", default=45, type=int)
    parser.add_argument("--sort_feat", type=int)
    args = parser.parse_args()

    # -- output dit -- #
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    writer = SummaryWriter(args.save_dir)

    # -- some tunable variables -- #
    device = 'cuda'
    epochs = 1000
    batch_size = 256

    fname = os.path.join(args.save_dir, "summary_{}.txt".format(args.tag))
    with open(fname, "w") as f:
        f.write("10-fold acc of the FNN trained on 135 N-sub variables, plus jet mass and multiplicity.\n")
        f.write("batch size: {}\n".format(batch_size))
        f.write("learning rate: {}\n".format("1e-4"))
        f.write("epochs: {}\n".format(epochs))
        f.write("Net layers: [135-800-800-800-800-800-64-7]\n")
        f.write("Dropout: 0.3\n")
        f.write("Hidden act: ReLU\n")
        f.write("Optimizer: Adam\n")
        f.write("Loss: CrossEntropyLoss\n")
        f.write("*****************************************\n")

        accuracy = []  # avg over all folds
        class_accuracy = []  # class avg over all folds
        mass_accuracy = []  # mass-bin avg over all folds
        pT_accuracy = []  # pT-bin avg over all folds
        for fold_id in range(10):
            print("*******************************************")
            print("* Nsubs Fold #: {}                         *".format(fold_id))
            print("*******************************************")
            # -- read and split the data -- #
            y, X, mass, pT = get_nsub_mass_dataset(nsubs_k=args.k)
            nsamples = y.shape[0]
            nclasses = len(np.unique(y))
            print("Total of {} samples with {} classes".format(nsamples, nclasses))
            print("No. of features: {}".format(X.shape[1]))

            data_train, data_val, data_test = split_dataset(X, y, mass, pT,
                                                            fold_id=fold_id,
                                                            num_folds=10)
            # # replace observable sort_feat with random instances of the
            # # observable taken form the training set
            rand_ix = np.arange(len(data_train))
            np.random.shuffle(rand_ix)
            rand_ix = rand_ix[:len(data_test)]
            X_test_shuff = data_test.X[:len(data_test)]
            X_test_shuff[:, args.sort_feat] = data_train.X[rand_ix, args.sort_feat]
            data_test_shuff = Dataset(X_test_shuff,
                                      data_test.y[:len(data_test)],
                                      data_test.mass[:len(data_test)],
                                      data_test.pT[:len(data_test)])

            generator = load_generator(data_train, data_val,
                                       data_test_shuff, batch_size=batch_size)

            # -- setup the model -- #
            hlnet_base = make_hlnet_base(input_dim=X.shape[1],
                                         inter_dim=800,
                                         num_hidden=5,
                                         out_dim=64,
                                         do_rate=0.3)
            model = HLNet(hlnet_base, out_dim=64, num_labels=nclasses).to(device)
            model = nn.DataParallel(model)
            print(model)
            print("Torch version: ", torch.__version__)
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            f.write("Num. of trainable params: {} \n".format(count_parameters(model)))
            print("Total no. of params: ", count_parameters(model))

            model_tag = "{}_{}".format(args.tag, fold_id)
            # for k=45 
            #hl_save_dir = "/home/alex/Desktop/Nprong/ml_models/nsubs_mass_fullrun"
            #hl_model_tag = "nsubs_mass_{}".format(fold_id)
            # for k=25 
            hl_save_dir = "/home/alex/Desktop/Nprong/ml_models/nsubs_mass_k25_fullrun"
            hl_model_tag = "nsubs_mass_{}".format(fold_id)
            acc, class_acc, mass_acc, pT_acc = main(model, hl_save_dir,
                                                    hl_model_tag, epochs)
            accuracy.append(acc)
            class_accuracy.append(class_acc)
            mass_accuracy.append(mass_acc)
            pT_accuracy.append(pT_acc)

            f.write("Fold {}\n".format(fold_id))
            f.write("Accuracy : \n")
            f.write(str(acc)+"\n")
            f.write("Class-bin accuracy : \n")
            f.write(str(class_acc)+"\n")
            f.write("Mass-bin accuracy : \n")
            f.write(str(mass_acc)+"\n")
            f.write("pT-bin accuracy : \n")
            f.write(str(pT_acc)+"\n")

        mean_accuracy = np.mean(accuracy)
        std_accuracy =  np.std(accuracy, ddof=1)
        mean_class_accuracy = np.mean(class_accuracy, axis=0)
        std_class_accuracy = np.std(class_accuracy, ddof=1, axis=0)
        mean_massbin_accuracy = np.mean(mass_accuracy, axis=0)
        std_massbin_accuracy = np.std(mass_accuracy, ddof=1, axis=0)
        mean_pTbin_accuracy = np.mean(pT_accuracy, axis=0)
        std_pTbin_accuracy = np.std(pT_accuracy, ddof=1, axis=0)

        f.write("*****************************************\n")
        f.write("Avg accuracy: \n")
        f.write(str(mean_accuracy) + " + " + str(std_accuracy) + "\n")
        f.write("Avg class-bin accuracy: \n")
        class_labels = ["N=1", "N=2", "N=3", "N=4b", "N=6", "N=8", "N=4q"]
        for st in zip(class_labels, mean_class_accuracy, std_class_accuracy):
            f.write(st[0] + " : " + str(st[1]) + " + " + str(st[2]) + "\n")
        mass_labels = ["[{}, {}]".format(300+i*50, 300+(i+1)*50) for i in range(8)]
        f.write("Avg mass-bin accuracy: \n")
        for st in zip(mass_labels, mean_massbin_accuracy, std_massbin_accuracy):
            f.write(st[0] + " : " + str(st[1]) + " + " + str(st[2]) + "\n")
        pT_labels = ["[{}, {}]".format(1000+i*25, 1000+(i+1)*25) for i in range(8)]
        f.write("Avg pT-bin accuracy: \n")
        for st in zip(pT_labels, mean_pTbin_accuracy, std_pTbin_accuracy):
            f.write(st[0] + " : " + str(st[1]) + " + " + str(st[2]) + "\n")
        f.write("*****************************************\n")

    # and also save to an h5 file
    fname_h5 = os.path.join(args.save_dir, "summary_{}.h5".format(args.tag))
    f_h5 = h5py.File(fname_h5, 'w')
    f_h5.create_dataset("mean_acc", data=mean_accuracy)
    f_h5.create_dataset("std_acc", data=std_accuracy)
    f_h5.create_group("per_prong")
    for i, nclass in enumerate(class_labels):
        f_h5.create_dataset("per_prong/{}_acc".format(nclass),
                             data=mean_class_accuracy[i])
        f_h5.create_dataset("per_prong/{}_std".format(nclass),
                             data=std_class_accuracy[i])
    f_h5.create_group("per_mass")
    mass_labels_low = ["{}GeV".format(300+i*50) for i in range(8)]
    for i, massbin in enumerate(mass_labels_low):
        f_h5.create_dataset("per_mass/{}_acc".format(massbin),
                             data=mean_massbin_accuracy[i])
        f_h5.create_dataset("per_mass/{}_std".format(massbin),
                             data=std_massbin_accuracy[i])
    f_h5.create_group("per_pT")
    pT_labels_low = ["{}GeV".format(1000+i*25) for i in range(8)]
    for i, pTbin in enumerate(pT_labels_low):
        f_h5.create_dataset("per_pT/{}_acc".format(pTbin),
                             data=mean_pTbin_accuracy[i])
        f_h5.create_dataset("per_pT/{}_std".format(pTbin),
                             data=std_pTbin_accuracy[i])
    f_h5.close()

    print("Done :)")
