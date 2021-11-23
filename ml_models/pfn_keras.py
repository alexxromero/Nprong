import os
import sys
import argparse
import time
import h5py
import numpy as np
from data_utils import plot_loss_acc
from data_utils import get_acc_per_class, get_acc_per_massbin, get_acc_per_pTbin
from data_utils import get_threeM_dataset, split_dataset

import energyflow as ef
from energyflow.archs import PFN

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

def setup_PFN(input_dim, lr, nclasses):
    opt = tf.keras.optimizers.Adam(lr=lr)
    model = PFN(input_dim=input_dim,
                output_dim=nclasses,
                Phi_sizes=(128, 128),
                Phi_acts='relu',
                F_sizes=(1024, 1024),
                F_acts='relu',
                F_dropouts=0.2,
                compile_opts={'optimizer': opt,
                              'loss': 'categorical_crossentropy',
                              'metrics': ['acc']})
    return model


def train_pfn(model, total_epochs, save_dir, model_tag, batch_size,
              data_train, data_val=None):
    best_model_path = os.path.join(save_dir, "pfn_{}_best.pt".format(model_tag))
    n_train = len(data_train)
    if data_val is not None:
        n_val = len(data_val)
        validation_data = (data_val[:n_val][0], data_val[:n_val][1])
    callbacks = [tf.keras.callbacks.ModelCheckpoint(
        best_model_path, monitor='val_acc', save_best_only=True)]
    history = model.fit(data_train[:n_train][0], data_train[:n_train][1],
                        epochs=total_epochs,
                        batch_size=batch_size,
                        validation_data=validation_data,
                        verbose=2,
                        callbacks=callbacks)
    plot_loss_acc(history.history['loss'],
                  history.history['acc'],
                  history.history['val_loss'],
                  history.history['val_acc'],
                  save_dir, model_tag)


def eval_pfn(model, save_dir, model_tag, summary_file, batch_size, data_test):
    best_model_path = os.path.join(save_dir, "pfn_{}_best.pt".format(model_tag))
    best_model = tf.keras.models.load_model(best_model_path)
    n_test = len(data_test)
    ypred_vect = best_model.predict(data_test[:n_test][0])
    ypred = np.argmax(ypred_vect, axis=1)
    ytrue = np.argmax(data_test[:n_test][1], axis=1)
    test_acc = (ypred == ytrue).sum().item() / ytrue.shape[0]
    class_acc = get_acc_per_class(ypred, ytrue)
    mass_acc = get_acc_per_massbin(ypred, ytrue, data_test[:n_test][2])
    pT_acc = get_acc_per_pTbin(ypred, ytrue, data_test[:n_test][3])
    f = open(summary_file, "a")
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
        save_dir, "pfn_eval_{}.h5".format(model_tag))
    fh5 = h5py.File(fh5_path, 'w')
    fh5.create_dataset("overall_acc", data=test_acc)
    fh5.create_dataset("class_accs", data=class_acc)
    fh5.create_dataset("mass_accs", data=mass_acc)
    fh5.create_dataset("pT_accs", data=pT_acc)
    fh5.create_dataset("pred_y", data=ypred)
    fh5.create_dataset("pred_y_vect", data=ypred_vect)
    fh5.create_dataset("true_y", data=ytrue)
    fh5.create_dataset("pT", data=data_test[:n_test][3])
    fh5.create_dataset("mass", data=data_test[:n_test][2])
    fh5.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nprong Model -- PFN')
    parser.add_argument("--state", type=str, choices=['train', 'test'])
    parser.add_argument("--fold", type=int)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--tag", type=str)
    args = parser.parse_args()

    # -- output dir -- #
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # -- some tunable variables -- #
    device = 'cuda'
    lr = 1e-4
    total_epochs = 1000
    batch_size = 256

    # -- get the jet constituents -- #
    y, threeM, mass, pT = get_threeM_dataset()
    nsamples = y.shape[0]
    nclasses = len(np.unique(y))

    fname = os.path.join(
        args.save_dir, "summary_{}_f{}.txt".format(args.tag, args.fold))
    f = open(fname, "w")
    f.write("Fold acc of the PFN model.\n")
    f.write("Fold: {}\n".format(args.fold))
    f.write("batch size: {}\n".format(batch_size))
    f.write("learning rate: {}\n".format(lr))
    f.write("epochs: {}\n".format(total_epochs))
    f.write("Phi layers: [128 128]\n")
    f.write("F layers: [1024 1024]\n")
    f.write("F_dropouts: 0.2\n")
    f.write("Optimizer: Adam\n")
    f.write("Output act: Softmax\n")
    f.write("Hiden act: ReLu\n")
    f.write("Loss: CrossEntropyLoss\n")
    f.write("*****************************************\n")

    print("*******************************************")
    print("* Fold #: {}                               *".format(args.fold))
    print("* Total of {} samples with {} classes  *".format(nsamples, nclasses))
    print("*******************************************")

    # -- split the data -- #
    y = tf.keras.utils.to_categorical(y, num_classes=nclasses)
    data_train, data_val, data_test = split_dataset(threeM, y, mass, pT,
                                                    fold_id=args.fold,
                                                    num_folds=10)

    # -- setup the model and print the no. of params -- #
    model = setup_PFN(input_dim=3, lr=lr, nclasses=nclasses)
    def count_parameters(model):
        return np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    f.write("Num. of trainable params: {} \n".format(count_parameters(model)))
    f.close()
    print("Total no. of params: ", count_parameters(model))

    # -- and execute the desired state -- #
    model_tag = "{}_f{}".format(args.tag, args.fold)
    if args.state == "train":
        train_pfn(
            model, total_epochs, args.save_dir, model_tag,
            batch_size=batch_size, data_train=data_train, data_val=data_val)
    elif args.state == "test":
        eval_pfn(
            model, args.save_dir, model_tag,
            summary_file=fname, batch_size=batch_size, data_test=data_test)

    print("Done :)")
