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

# using the same architecture as the nsubs
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nprong Model -- DNN')
    parser.add_argument("--state", type=str, choices=['train', 'test'])
    parser.add_argument("--fold", type=int)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--tag", type=str)
    parser.add_argument("--resume_from_epoch", type=int, default=None)
    args = parser.parse_args()

    # -- output dir -- #
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # -- some tunable variables -- #
    device = 'cuda'
    total_epochs = 1000
    lr = 1e-4
    batch_size = 256

    # -- get the jet observables -- #
    y, X, mass, pT = get_nsub_EFP_mass_multi_dataset()
    lasso_ix = [ 45,  91, 178, 137,  48, 297,  50, 257,   4,  46,
                  1,  54,  96,  94,  12, 254,  20, 248, 298,   2,
                266, 204, 261,  47, 140, 237,  92, 268, 155, 247,
                193]
    X = X[:, lasso_ix]
    print("No. features: ", X.shape[1])
    nsamples = y.shape[0]
    nclasses = len(np.unique(y))

    fname = os.path.join(
        args.save_dir, "summary_{}_f{}.txt".format(args.tag, args.fold))
    f = open(fname, "w")
    f.write("Fold acc of the DNN model trained on mass+nsubs.\n")
    f.write("Fold: {}\n".format(args.fold))
    f.write("batch size: {}\n".format(batch_size))
    f.write("learning rate: {}\n".format(lr))
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
    model = HLNet(hlnet_base, out_dim=64, num_labels=nclasses).to(device)
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
        train_torch_model(device=device, model=model, generator=generator,
                          optimizer=optimizer, loss_fn=loss_fn,
                          save_dir=args.save_dir, model_tag=model_tag,
                          total_epochs=total_epochs,
                          resume_from_epoch=args.resume_from_epoch)
    elif args.state == "test":
        eval_torch_model(device=device, model=model, generator=generator,
                         loss_fn=loss_fn,
                         save_dir=args.save_dir, model_tag=model_tag,
                         summary_file=fname)

    print("Done :)")
