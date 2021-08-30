# Based on Yadong's networks. Similar code, I just cleaned up the
# part of the analysis that are no longer needed.

import os
import sys
import argparse
import time
import h5py
import numpy as np
from utils import get_nsub_EFP_mass_multi_dataset, split_dataset
from utils import plot_loss_acc
from utils import get_acc_per_class, get_acc_per_massbin, get_acc_per_pTbin

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

class GatedHLNet(nn.Module):
    def __init__(self, hlnet_base, num_labels=7):
        super(GatedHLNet, self).__init__()
        self.gates = nn.Parameter(data=torch.randn(299))
        self.hlnet_base = hlnet_base
        self.top = nn.Linear(128, num_labels)
        self.bn = nn.BatchNorm1d(64)

    def path(self, X, gates):
        X = gates * X
        X = self.hlnet_base(X)
        X = self.bn(X)
        out = self.top(X)
        return out

    def forward(self, X):
        out = self.path(X, self.gates)
        if model.training:
            return out, self.gates
        else:
            gates_clipped = torch.where(self.gates.abs() > 1e-2,
                                        self.gates,
                                        torch.zeros_like(self.gates))
            out_clipped = self.path(X, gates_clipped)
            return out, out_clipped, self.gates


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


def train(model, train_generator, optimizer, loss_fn, epoch, val_generator=None):
    model.train()
    train_loss, train_acc = 0.0, 0.0
    iterations = len(train_generator)
    for i, (X, y, mass, pT) in enumerate(train_generator):
        optimizer.zero_grad()

        X = torch.tensor(X).float().to(device)
        y = torch.tensor(y).long().to(device)

        ypred, gates = model(X)
        loss = loss_fn(ypred, y)
        gate_loss = strength * torch.abs(gates).mean()
        loss = loss.mean() + gate_loss
        loss.backward()
        optimizer.step()
        acc = (torch.argmax(ypred, dim=1) == y).sum().item() / y.shape[0]

        train_loss += loss.item()
        train_acc += acc

        if i % 50 == 0:
            print('Train loss: ', loss.item(), '  Train acc: ', acc)
            writer.add_scalar('Loss/train', loss.item(), epoch * iterations + i)
            writer.add_scalar('Acc/train', acc, epoch * iterations + i)

    train_loss /= len(train_generator)  # avg over all batches in the epoch
    train_acc  /= len(train_generator)  # avg over all batches in the epoch

    if val_generator is not None:
        val_loss, val_acc, val_acc_clipped, predy_truey_mass_pT, gates\
            = test(model, val_generator, loss_fn=loss_fn, epoch=epoch, validate=True)
        print('Val loss: ', val_loss, ' Val acc: ', val_acc)
        return train_loss, train_acc, val_loss, val_acc, val_acc_clipped, model
    return train_loss, train_acc, model


def test(model, generator, loss_fn, epoch=None, validate=False):
    model.eval()
    predy_truey_mass_pT = []
    test_loss, test_acc, test_acc_clipped = 0.0, 0.0, 0.0
    nsamples = 0
    gates = model.module.gates if isinstance(model, nn.DataParallel) else model.gates

    with torch.no_grad():
        for i, (X, y, mass, pT) in enumerate(generator):
            X = torch.tensor(X).float().to(device)
            y = torch.tensor(y).long().to(device)

            ypred, ypred_clipped = model(X)
            loss = loss_fn(ypred, y)

            pred_y_clipped = torch.argmax(pred_clipped, dim=1)
            pred_y = torch.argmax(ypred, dim=1)
            predy_truey_mass_pT.append(torch.stack(
                [pred_y.float().cpu(), pred_y_clipped.float().cpu(), y.float().cpu(), mass.float(), pT.float()])
                )

            test_loss += loss.sum().item()
            test_acc  += (torch.argmax(ypred, dim=1) == y).sum().item()
            test_acc_clipped += (torch.argmax(ypred_clipped, dim=1) == y).sum().item()
            nsamples += y.shape[0]

    gates = gates.detach().cpu().numpy()
    num_remaining_feat = (np.abs(gates) > 1e-2).sum().tolist()

    predy_truey_mass_pT = torch.cat(predy_truey_mass_pT, dim=1).numpy()
    test_loss /= nsamples  # avg over all samples
    test_acc  /= nsamples  # avg over all samples
    test_acc_clipped /= nsamples

    if validate:
        writer.add_scalar('Loss/val', test_loss, epoch)
        writer.add_scalar('Acc/val', test_acc, epoch)
        writer.add_scalar('Acc_clipped/val', test_acc_clipped, epoch)
        writer.add_scalar('num_remaining_feat', num_remaining_feat, epoch)

    return test_loss, test_acc, test_acc_clipped, predy_truey_mass_pT, gates


def main(model, save_dir, model_tag):
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

    # -- begin training -- #
    best_acc = 0
    saved_best_model = os.path.join(save_dir,
                                    "best_model_{}.pt".format(model_tag))
    train_loss_ls, train_acc_ls, val_loss_ls, val_acc_ls = [], [], [], []
    for i in range(epochs):
        print('Starting epoch ', i)
        train_loss, train_acc, val_loss, val_acc, val_acc_clipped, model = train(
            model, generator['train'], optimizer, loss_fn, epoch=i,
            val_generator=generator['val']
            )
        if val_acc_clipped > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), saved_best_model)
            print("model saved")
        if (i+1) % 100 == 0:
            torch.save(model.state_dict(), "./dummy_{}.pt".format(i))
            print("model saved at epoch ", i)
        train_loss_ls.append(train_loss)
        train_acc_ls.append(train_acc)
        val_loss_ls.append(val_loss)
        val_acc_ls.append(val_acc)
    writer.close()

    print('Loading best model...')
    cp = torch.load(saved_best_model)
    model.load_state_dict(cp, strict=True)

    test_loss, test_acc, test_acc_clipped, predy_truey_mass_pT, gates = \
        test(model, generator['test'], loss_fn)
    print("Total Acc: ", test_acc)

    # -- analyze gates ------------------------------------------ # 
    # ----------------------------------------------------------- #
    gates = model.module.gates.detach().cpu().numpy()
    gates_selected_i = np.where(np.abs(gates) > 1e-2)[0]
    gates_selected_v = gates[gates_selected_i]
    gates_ascend = np.argsort(gates_selected_v)[::-1]
    num_remaining_feat = len(gates_selected_i)
    print('gates > 0.01, ascending order: ')
    print(gates_selected_i[gates_ascend])
    # ----------------------------------------------------------- #

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

    plot_loss_acc(train_loss_ls, train_acc_ls, val_loss_ls, val_acc_ls,
                  save_dir, model_tag)

    return test_acc, class_acc, mass_acc, pT_acc


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Nprong Model')
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--tag", type=str)
    args = parser.parse_args()

    # -- output dit -- #
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    writer = SummaryWriter(args.save_dir)

    # -- some tunable variables -- #
    device = 'cuda'
    lr = 1e-4  # or 1e-4
    epochs = 1000
    batch_size = 256
    #fold_id = None  # doing 10 bootstraps now

    fname = os.path.join(args.save_dir, "summary_{}.txt".format(args.tag))
    with open(fname, "w") as f:
        f.write("10-fold acc of the FNN trained on 135 N-sub variables plus IRC-safe EFPs (and mass+multi).\n")
        f.write("LASSO is used to narrow down the features.")
        f.write("batch size: {}\n".format(batch_size))
        f.write("learning rate: {}\n".format(lr))
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
            y, X, mass, pT = get_nsub_EFP_mass_multi_dataset(args.input_file)
            print("No. features: ", X.shape[1])
            nsamples = y.shape[0]
            nclasses = len(np.unique(y))
            print("Total of {} samples with {} classes".format(nsamples, nclasses))

            data_train, data_val, data_test = split_dataset(X, y, mass, pT,
                                                            fold_id=fold_id,
                                                            num_folds=10)
            generator = load_generator(data_train, data_val,
                                       data_test, batch_size=batch_size)

            # -- setup the model -- #
            hlnet_base = make_hlnet_base(input_dim=X.shape[1],
                                         inter_dim=800,
                                         num_hidden=5,
                                         out_dim=64,
                                         do_rate=0.3)
            model = GatedHLNet(hlnet_base, num_labels=nclasses).to(device)
            model = nn.DataParallel(model)
            print(model)
            print("Torch version: ", torch.__version__)
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            f.write("Num. of trainable params: {} \n".format(count_parameters(model)))
            print("Total no. of params: ", count_parameters(model))

            model_tag = "{}_{}".format(args.tag, fold_id)
            acc, class_acc, mass_acc, pT_acc = main(model, args.save_dir,
                                                    model_tag)
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
        class_labels = ["N=1 ", "N=2 ", "N=3 ", "N=4b", "N=8", "N=4q", "N=6"]
        for st in zip(class_labels, mean_class_accuracy, std_class_accuracy):
            f.write(st[0] + " : " + str(st[1]) + " + " + str(st[2]) + "\n")
        mass_labels = ["[{}, {}]".format(300+i*50, 300+(i+1)*50) for i in range(8)]
        f.write("Avg mass-bin accuracy: \n")
        for st in zip(mass_labels, mean_massbin_accuracy, std_massbin_accuracy):
            f.write(st[0] + " : " + str(st[1]) + " + " + str(st[2]) + "\n")
        pT_labels = ["[{}, {}]".format(1000+i*20, 1000+(i+1)*20) for i in range(11)]
        f.write("Avg pT-bin accuracy: \n")
        for st in zip(pT_labels, mean_pTbin_accuracy, std_pTbin_accuracy):
            f.write(st[0] + " : " + str(st[1]) + " + " + str(st[2]) + "\n")
        f.write("*****************************************\n")

    print("Done :)")