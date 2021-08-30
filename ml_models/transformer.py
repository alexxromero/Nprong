# Based on Yadong's networks. Similar code, I just cleaned up the
# part of the analysis that are no longer needed.

import os
import sys
import argparse
import time
import h5py
import numpy as np
from utils import get_threeM_dataset, split_dataset
from utils import plot_loss_acc
from utils import get_acc_per_class, get_acc_per_massbin, get_acc_per_pTbin

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
#from utils import get_batch_classwise_acc

import transformers 
from transformers import BertModel, BertConfig
from bert import BertForSequenceClassification

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

config = BertConfig(hidden_size=256,
                    num_hidden_layers=4,
                    num_attention_heads=8,
                    intermediate_size=128,
                    num_labels=7,
                    input_dim=230,
                    attention_probs_dropout_prob=0.1,
                    hidden_dropout_prob=0.1)

def train(model, train_generator, optimizer, loss_fn, epoch, val_generator=None):
    model.train()
    train_loss, train_acc = 0.0, 0.0
    iterations = len(train_generator)
    for i, (nsubs, y, mass, pT) in enumerate(train_generator):
        optimizer.zero_grad()

        nsubs = torch.tensor(nsubs).float().to(device)
        y = torch.tensor(y).long().to(device)

        ypred = model(nsubs)
        loss = loss_fn(ypred, y)
        loss = loss.mean()
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
        val_loss, val_acc, _ = test(model, val_generator,
                                    loss_fn=loss_fn, epoch=epoch,
                                    validate=True)
        print('Val loss: ', val_loss, ' Val acc: ', val_acc)
        return train_loss, train_acc, val_loss, val_acc, model
    return train_loss, train_acc, model


def test(model, generator, loss_fn, epoch=None, validate=False):
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


def main(model, save_dir, model_tag):

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 400, 600, 800], gamma=0.5, last_epoch=-1)
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    # -- begin training -- #
    best_acc = 0
    saved_best_model = os.path.join(save_dir,
                                    "best_model_{}.pt".format(model_tag))
    train_loss_ls, train_acc_ls, val_loss_ls, val_acc_ls = [], [], [], []
    for i in range(epochs):
        print('Starting epoch ', i)
        train_loss, train_acc, val_loss, val_acc, model = train(
            model, generator['train'], optimizer, loss_fn, epoch=i,
            val_generator=generator['val']
            )
        if val_acc > best_acc:
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

    _, test_acc, predy_truey_mass_pT = test(model, generator['test'], loss_fn)
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

    plot_loss_acc(train_loss_ls, train_acc_ls, val_loss_ls, val_acc_ls,
                  save_dir, model_tag)

    return test_acc, class_acc, mass_acc, pT_acc



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Nprong Model -- Bert')
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
    lr = 1e-4  # or 1e-3
    epochs = 1000
    batch_size = 256
    #fold_id = None  # doing 10 bootstraps now

    fname = os.path.join(args.save_dir, "summary_{}.txt".format(args.tag))
    with open(fname, "w") as f:
        f.write("10-fold acc of the Transformer.\n")
        f.write("batch size: {}\n".format(batch_size))
        f.write("learning rate: {}\n".format(lr))
        f.write("epochs: {}\n".format(epochs))
        f.write("Optimizer: Adam\n")
        f.write("Output act: Softmax")
        f.write("Loss: CrossEntropyLoss\n")

        f.write("*****************************************\n")

        accuracy = []  # avg over all folds
        class_accuracy = []  # class avg over all folds
        mass_accuracy = []  # mass-bin avg over all folds
        pT_accuracy = []  # pT-bin avg over all folds
        for fold_id in range(10):
            print("*******************************************")
            print("* PFN Fold #: {}                           *".format(fold_id))
            print("*******************************************")
            # -- read and split the data -- #
            y, threeM, mass, pT = get_threeM_dataset(args.input_file)
            nsamples = y.shape[0]
            nclasses = len(np.unique(y))
            print("Total of {} samples with {} classes".format(nsamples, nclasses))

            data_train, data_val, data_test = split_dataset(threeM, y, mass, pT,
                                                            fold_id=fold_id,
                                                            num_folds=10)
            generator = load_generator(data_train, data_val,
                                       data_test, batch_size=batch_size)

            # -- setup the model -- #
            model = BertForSequenceClassification(config).to(device)
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            model = nn.DataParallel(model)
            print(model)
            print("Torch version: ", torch.__version__)
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
