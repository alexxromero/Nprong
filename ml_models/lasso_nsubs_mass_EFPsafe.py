# Based on Yadong's networks. Similar code, I just cleaned up the
# part of the analysis that are no longer needed.

import os
import sys
import argparse
import time
import h5py
import numpy as np
from data_utils import get_nsub_EFP_mass_multi_dataset, split_dataset
from data_utils import plot_loss_acc
from data_utils import get_acc_per_class, get_acc_per_massbin, get_acc_per_pTbin

# using the same base architecture as the nsubs
from nsubs_mass import make_hlnet_base, HLNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
#from utils import get_batch_classwise_acc

from torch_utils import load_generator, train_torch_model, eval_torch_model

os.environ['CUDA_VISIBLE_DEVICES'] = torch.cuda.get_device_name(0)
print('training using GPU:', torch.cuda.get_device_name(0))

class GatedHLNet(nn.Module):
    def __init__(self, hlnet_base, nlabels=7, ngates=299):
        super(GatedHLNet, self).__init__()
        self.gates = nn.Parameter(data=torch.randn(ngates))
        self.hlnet_base = hlnet_base
        self.top = nn.Linear(64, nlabels)
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


def train(device, model, writer, train_generator, optimizer, loss_fn, epoch,
          val_generator=None, strength=10):
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
        val_loss, val_acc = test(
            device, model, val_generator, loss_fn, epoch,
            validate=True, writer=writer)
        print('Val loss: ', val_loss, ' Val acc: ', val_acc)
        return train_loss, train_acc, val_loss, val_acc, model
    return train_loss, train_acc, model


def test(device, model, generator, loss_fn, epoch=None, validate=False,
         writer=None):
    model.eval()
    predy_predyclip_truey_mass_pT = []
    test_loss, test_acc, test_acc_clipped = 0.0, 0.0, 0.0
    nsamples = 0
    with torch.no_grad():
        for i, (X, y, mass, pT) in enumerate(generator):
            X = torch.tensor(X).float().to(device)
            y = torch.tensor(y).long().to(device)
            ypred_vect, ypred_vect_clipped, _ = model(X)
            loss = loss_fn(ypred_vect, y)
            ypred = torch.argmax(ypred_vect, dim=1)
            ypred_clipped = torch.argmax(ypred_vect_clipped, dim=1)
            predy_predyclip_truey_mass_pT.append(torch.stack(
                [ypred.float().cpu(), ypred_clipped.float().cpu(),
                 y.float().cpu(), mass.float(), pT.float()]))
            test_loss += loss.sum().item()
            test_acc  += (ypred == y).sum().item()
            test_acc_clipped  += (ypred_clipped == y).sum().item()
            nsamples += y.shape[0]
    test_loss /= nsamples  # avg over all samples
    test_acc  /= nsamples  # avg over all samples
    test_acc_clipped  /= nsamples  # avg over all samples
    if validate and writer is not None:
        writer.add_scalar('Loss/val', test_loss, epoch)
        writer.add_scalar('Acc/val', test_acc, epoch)
        return test_loss, test_acc
    all_arrays = torch.cat(predy_predyclip_truey_mass_pT, dim=1).numpy()
    return test_loss, test_acc, test_acc_clipped, all_arrays


def train_gated_model(device, model, generator, optimizer, loss_fn,
                      save_dir, model_tag, total_epochs, strength=10):
    writer = SummaryWriter(save_dir)
    best_model_path = os.path.join(
        save_dir, "bert_{}_best.pt".format(model_tag))

    last_epoch = 0
    best_acc = 0
    train_loss_ls, train_acc_ls, val_loss_ls, val_acc_ls = [], [], [], []
    for epoch in range(total_epochs):
        print('Starting epoch ', epoch)
        train_loss, train_acc, val_loss, val_acc, model = train(
            device, model, writer, generator['train'], optimizer, loss_fn,
            epoch, generator['val'], strength=strength)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print("best model saved")
        if (epoch + 1) % 50 == 0:
            checkpoint_path = os.path.join(save_dir,
            "bert_{}_e{}.pt".format(model_tag, epoch))
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss_ls': train_loss_ls,
                        'train_acc_ls': train_acc_ls,
                        'val_loss_ls': val_loss_ls,
                        'val_acc_ls': val_acc_ls,
                        'best_acc': best_acc},
                        checkpoint_path)
        train_loss_ls.append(train_loss)
        train_acc_ls.append(train_acc)
        val_loss_ls.append(val_loss)
        val_acc_ls.append(val_acc)
    plot_loss_acc(train_loss_ls, train_acc_ls, val_loss_ls, val_acc_ls,
                  save_dir, model_tag)
    writer.close()


def eval_gated_model(device, model, generator, loss_fn,
                     save_dir, model_tag, summary_file):
    f = open(summary_file, "a")
    print('Loading best model...')
    best_model_path = os.path.join(
        save_dir, "bert_{}_best.pt".format(model_tag))
    cp = torch.load(best_model_path)
    model.load_state_dict(cp, strict=True)
    _, test_acc, test_acc_clip, predy_predyclip_truey_mass_pT = \
        test(device, model, generator['test'], loss_fn)
    print("Total Acc: ", test_acc)
    print("Total Acc -- clipped gates: ", test_acc_clip)
    # -- evaluate the model -- #
    pred_y = predy_predyclip_truey_mass_pT[0, :]
    pred_y_clip = predy_predyclip_truey_mass_pT[1, :]
    true_y = predy_predyclip_truey_mass_pT[2, :]
    mass = predy_predyclip_truey_mass_pT[3, :]
    pT = predy_predyclip_truey_mass_pT[4, :]
    class_acc = get_acc_per_class(pred_y, true_y)
    mass_acc = get_acc_per_massbin(pred_y, true_y, mass)
    pT_acc = get_acc_per_pTbin(pred_y, true_y, pT)
    class_acc_clipped = get_acc_per_class(pred_y_clip, true_y)
    mass_acc_clipped = get_acc_per_massbin(pred_y_clip, true_y, mass)
    pT_acc_clipped = get_acc_per_pTbin(pred_y_clip, true_y, pT)
    f.write("Accuracy : \n")
    f.write(str(test_acc)+"\n")
    f.write("Class-bin accuracy : \n")
    class_labels = ["N=1", "N=2", "N=3", "N=4b", "N=6", "N=8", "N=4q"]
    for st in zip(class_labels, class_acc):
        f.write(st[0] + " : " + str(st[1]) + "\n")
        print(st[0] + " : " + str(st[1]) + "\n")
    f.write("Mass-bin accuracy : \n")
    f.write(str(mass_acc)+"\n")
    f.write("pT-bin accuracy : \n")
    f.write(str(pT_acc)+"\n")
    f.write("\n")
    f.write("Accuracy -- clipped : \n")
    f.write(str(test_acc_clip)+"\n")
    f.write("Class-bin accuracy -- clipped : \n")
    class_labels = ["N=1", "N=2", "N=3", "N=4b", "N=6", "N=8", "N=4q"]
    for st in zip(class_labels, class_acc_clipped):
        f.write(st[0] + " : " + str(st[1]) + "\n")
    f.write("Mass-bin accuracy -- clipped : \n")
    f.write(str(mass_acc_clipped)+"\n")
    f.write("pT-bin accuracy -- clipped : \n")
    f.write(str(pT_acc_clipped)+"\n")
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nprong Model -- DNN')
    parser.add_argument("--state", type=str, choices=['train', 'test'])
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--tag", type=str)
    args = parser.parse_args()

    # -- output dir -- #
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # -- some tunable variables -- #
    device = 'cuda'
    total_epochs = 1000
    lr = 1e-4
    batch_size = 256
    strength = 10

    # -- get the jet observables -- #
    y, X, mass, pT = get_nsub_EFP_mass_multi_dataset()
    nsamples = y.shape[0]
    nclasses = len(np.unique(y))

    fname = os.path.join(
        args.save_dir, "summary_{}_f{}.txt".format(args.tag, args.fold))
    f = open(fname, "w")
    f.write("Fold acc of the DNN model trained on mass+nsubs.\n")
    f.write("Fold: {}\n".format(args.fold))
    f.write("batch size: {}\n".format(batch_size))
    f.write("learning rate: {}\n".format(lr))
    f.write("strength: {}\n".format(strength))
    f.write("epochs: {}\n".format(total_epochs))
    f.write("*****************************************\n")

    print("*******************************************")
    print("* Fold #: {}                               *".format(args.fold))
    print("* Total of {} samples with {} classes  *".format(nsamples, nclasses))
    print("*******************************************")

    # -- split the data -- #
    data_train, data_val, data_test = split_dataset(X, y, mass, pT,
                                                    fold_id=args.fold,
                                                    num_folds=10)
    generator = load_generator(data_train, data_val, data_test,
                               batch_size=batch_size)

    # -- setup the model -- #
    hlnet_base = make_hlnet_base(input_dim=X.shape[1],
                                 inter_dim=800,
                                 num_hidden=5,
                                 out_dim=64,
                                 do_rate=0.3)
    model = GatedHLNet(
        hlnet_base, nlabels=nclasses, ngates=X.shape[1]).to(device)
    model = nn.DataParallel(model)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    f.write("Num. of trainable params: {} \n".format(count_parameters(model)))
    f.close()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[200, 400, 600, 800], gamma=0.5, last_epoch=-1)
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    model_tag = "{}_f{}".format(args.tag, args.fold)

    # -- and execute the desired state -- #
    if args.state == "train":
        train_gated_model(device=device, model=model, generator=generator,
                          optimizer=optimizer, loss_fn=loss_fn,
                          save_dir=args.save_dir, model_tag=model_tag,
                          total_epochs=total_epochs, strength=strength)
        gates = model.module.gates.detach().cpu().numpy()
        gates_ix = np.where(np.abs(gates) > 1e-2)[0]
        gates_vals = gates[gates_ix]
        gates_argsort = np.argsort(gates_vals)[::-1]
        gates_ix_sorted = gates_ix[gates_argsort]
        fname_games = os.path.join(
            args.save_dir, "gates_{}_f{}.txt".format(args.tag, args.fold))
        f = open(fname_games, "w")
        f.write("Total of {} selected gates \n".format(len(gates_ix)))
        f.write("Gates indices (in descending order of their magnitudes): \n")
        f.write(str(gates_ix_sorted))
        f.close()
        print("Selected gates: ")
        print(gates_ix_sorted)

    elif args.state == "test":
        eval_gated_model(device=device, model=model, generator=generator,
                         loss_fn=loss_fn,
                         save_dir=args.save_dir, model_tag=model_tag,
                         summary_file=fname)

    print("Done :)")
