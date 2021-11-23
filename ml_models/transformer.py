# Based on Yadong's networks. Similar code, I just cleaned up the
# part of the analysis that are no longer needed.

import os
import sys
import argparse
import time
import h5py
import numpy as np
from data_utils import get_threeM_dataset, split_dataset
from data_utils import plot_loss_acc
from data_utils import get_acc_per_class, get_acc_per_massbin, get_acc_per_pTbin

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
#from utils import get_batch_classwise_acc

from torch_utils import load_generator, train_torch_model, eval_torch_model

import transformers
from transformers import BertModel, BertConfig
from bert import BertForSequenceClassification

os.environ['CUDA_VISIBLE_DEVICES'] = torch.cuda.get_device_name(0)
print('training using GPU:', torch.cuda.get_device_name(0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nprong Model -- Bert')
    parser.add_argument("--state", type=str, choices=['train', 'test'])
    parser.add_argument("--fold", type=int)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--tag", type=str)
    parser.add_argument("--resume_from_epoch", type=str, default=None)
    args = parser.parse_args()

    # -- output dir -- #
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # -- some tunable parameters -- #
    device = 'cuda'
    total_epochs = 1000
    lr = 1e-4
    batch_size = 256

    # -- get the jet constituents -- #
    y, threeM, mass, pT = get_threeM_dataset()
    nsamples = y.shape[0]
    nclasses = len(np.unique(y))

    fname = os.path.join(args.save_dir,
                         "summary_{}_f{}.txt".format(args.tag, args.fold))
    f = open(fname, "w")
    f.write("Fold acc of the BERT model.\n")
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
    data_train, data_val, data_test = split_dataset(threeM, y, mass, pT,
                                                    fold_id=args.fold,
                                                    num_folds=10)
    generator = load_generator(data_train, data_val, data_test,
                               batch_size=batch_size)

    # -- setup the model -- #
    config = BertConfig(hidden_size=256,
                        num_hidden_layers=4,
                        num_attention_heads=8,
                        intermediate_size=128,
                        num_labels=7,
                        input_dim=230,
                        attention_probs_dropout_prob=0.1,
                        hidden_dropout_prob=0.1)
    model = BertForSequenceClassification(config).to(device)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = nn.DataParallel(model)
    print(model)
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
