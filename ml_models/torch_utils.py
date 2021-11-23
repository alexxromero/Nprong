import os
import sys
import argparse
import time
import h5py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from data_utils import plot_loss_acc
from data_utils import get_acc_per_class, get_acc_per_massbin, get_acc_per_pTbin


def load_generator(data_train, data_val, data_test, batch_size=256):
    generator = {}
    generator['train'] = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=True, num_workers=8)
    generator['val'] = torch.utils.data.DataLoader(
        data_val, batch_size=batch_size, shuffle=True, num_workers=8)
    generator['test'] = torch.utils.data.DataLoader(
        data_test, batch_size=batch_size, shuffle=True, num_workers=8)
    return generator


def train(device, model, writer, train_generator, optimizer, loss_fn, epoch,
          val_generator=None):
    model.train()
    train_loss, train_acc = 0.0, 0.0
    iterations = len(train_generator)
    for i, (X, y, mass, pT) in enumerate(train_generator):
        optimizer.zero_grad()
        X = torch.tensor(X).float().to(device)
        y = torch.tensor(y).long().to(device)
        ypred = model(X)
        loss = loss_fn(ypred, y).mean()
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
    predy_truey_mass_pT, predy_vector = [], []
    test_loss, test_acc = 0.0, 0.0
    nsamples = 0
    with torch.no_grad():
        for i, (X, y, mass, pT) in enumerate(generator):
            X = torch.tensor(X).float().to(device)
            y = torch.tensor(y).long().to(device)
            ypred_vect = model(X)
            loss = loss_fn(ypred_vect, y)
            ypred = torch.argmax(ypred_vect, dim=1)
            predy_truey_mass_pT.append(torch.stack(
                [ypred.float().cpu(), y.float().cpu(),
                 mass.float(), pT.float()]))
            predy_vector.append(ypred_vect.float().cpu().numpy())
            test_loss += loss.sum().item()
            test_acc  += (ypred == y).sum().item()
            nsamples += y.shape[0]
    test_loss /= nsamples  # avg over all samples
    test_acc  /= nsamples  # avg over all samples
    if validate and writer is not None:
        writer.add_scalar('Loss/val', test_loss, epoch)
        writer.add_scalar('Acc/val', test_acc, epoch)
        return test_loss, test_acc
    predy_vector = np.concatenate([v for v in predy_vector], axis=0)
    predy_truey_mass_pT = torch.cat(predy_truey_mass_pT, dim=1).numpy()
    return test_loss, test_acc, predy_truey_mass_pT, predy_vector


def train_torch_model(device, model, generator, optimizer, loss_fn,
                      save_dir, model_tag,
                      total_epochs, resume_from_epoch=None):
    writer = SummaryWriter(save_dir)
    best_model_path = os.path.join(
        save_dir, "bert_{}_best.pt".format(model_tag))
    if resume_from_epoch is not None:  # load model's status and resume training
        checkpoint_path = os.path.join(save_dir,
            "bert_{}_e{}.pt".format(model_tag, resume_from_epoch))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        train_loss_ls = checkpoint['train_loss_ls']
        train_acc_ls = checkpoint['train_acc_ls']
        val_loss_ls = checkpoint['val_loss_ls']
        val_acc_ls = checkpoint['val_acc_ls']
        best_acc = checkpoint['best_acc']
    else:
        last_epoch = 0
        best_acc = 0
        train_loss_ls, train_acc_ls, val_loss_ls, val_acc_ls = [], [], [], []
    for i in range(total_epochs):
        epoch = last_epoch + i
        if epoch >= total_epochs:
            break
        print('Starting epoch ', epoch)
        train_loss, train_acc, val_loss, val_acc, model = train(
            device, model, writer, generator['train'], optimizer, loss_fn,
            epoch, generator['val'])
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


def eval_torch_model(device, model, generator, loss_fn,
                     save_dir, model_tag, summary_file):
    f = open(summary_file, "a")
    print('Loading best model...')
    best_model_path = os.path.join(
        save_dir, "bert_{}_best.pt".format(model_tag))
    cp = torch.load(best_model_path)
    model.load_state_dict(cp, strict=True)
    _, test_acc, predy_truey_mass_pT, predy_vector = test(
        device, model, generator['test'], loss_fn)
    print("Total Acc: ", test_acc)
    # -- evaluate the model -- #
    pred_y = predy_truey_mass_pT[0, :]
    true_y = predy_truey_mass_pT[1, :]
    mass = predy_truey_mass_pT[2, :]
    pT = predy_truey_mass_pT[3, :]
    class_acc = get_acc_per_class(pred_y, true_y)
    mass_acc = get_acc_per_massbin(pred_y, true_y, mass)
    pT_acc = get_acc_per_pTbin(pred_y, true_y, pT)
    f.write("Accuracy : \n")
    f.write(str(test_acc)+"\n")
    f.write("Class-bin accuracy : \n")
    class_labels = ["N=1", "N=2", "N=3", "N=4b", "N=6", "N=8", "N=4q"]
    for st in zip(class_labels, class_acc):
        f.write(st[0] + " : " + str(st[1]) + "\n")
    f.write("Mass-bin accuracy : \n")
    f.write(str(mass_acc)+"\n")
    f.write("pT-bin accuracy : \n")
    f.write(str(pT_acc)+"\n")
    f.close()
    # -- and save the predictions to an h5 file -- #
    fh5_path = os.path.join(
        save_dir, "bert_eval_{}.h5".format(model_tag))
    fh5 = h5py.File(fh5_path, 'w')
    fh5.create_dataset("overall_acc", data=test_acc)
    fh5.create_dataset("class_accs", data=class_acc)
    fh5.create_dataset("mass_accs", data=mass_acc)
    fh5.create_dataset("pT_accs", data=pT_acc)
    fh5.create_dataset("pred_y", data=pred_y)
    fh5.create_dataset("pred_y_vect", data=predy_vector)
    fh5.create_dataset("true_y", data=true_y)
    fh5.create_dataset("pT", data=pT)
    fh5.create_dataset("mass", data=mass)
    fh5.close()
