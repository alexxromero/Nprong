import os
import sys
import argparse
import time
import h5py
import numpy as np
from utils import plot_loss_acc
from utils import get_acc_per_class, get_acc_per_massbin, get_acc_per_pTbin
from utils import get_threeM_dataset, split_dataset

import energyflow as ef
from energyflow.archs import EFN

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

# os.environ['CUDA_VISIBLE_DEVICES'] = torch.cuda.get_device_name(0)
# print('training using GPU:', torch.cuda.get_device_name(0))

def setup_EFN(input_dim, lr):
    opt = tf.keras.optimizers.Adam(lr=lr)
    model = EFN(input_dim=input_dim,
                output_dim=nclasses,
                Phi_sizes=(128, 128),
                Phi_acts = 'relu',
                F_sizes=(1024, 1024),
                F_acts = 'relu',
                F_dropouts=0.2,
                compile_opts={'optimizer': opt,
                              'loss': 'categorical_crossentropy',
                              'metrics': ['acc']})
    return model


def train_model(model, save_dir, model_tag, data_train, data_val):
    saved_best_model = os.path.join(save_dir,
                                    "best_model_{}.h5".format(model_tag))
    callbacks = [tf.keras.callbacks.ModelCheckpoint(
        saved_best_model, monitor='val_acc', save_best_only=True)]

    n_train = len(data_train)
    n_val = len(data_val)
    X_train = [data_train[:n_train][0][:, :, 0], data_train[:n_train][0][:, :, 1:]]
    X_val = [data_val[:n_train][0][:, :, 0], data_val[:n_train][0][:, :, 1:]]
    history = model.fit(X_train, data_train[:n_train][1],
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_val,
                                         data_val[:n_val][1]),
                        callbacks=callbacks)
    plot_loss_acc(history.history['loss'],
                  history.history['acc'],
                  history.history['val_loss'],
                  history.history['val_acc'],
                  save_dir, model_tag)


def eval_model(save_dir, model_tag, data_test):
    saved_best_model = os.path.join(save_dir,
                                    "best_model_{}.h5".format(model_tag))
    model = tf.keras.models.load_model(saved_best_model)
    n_test = len(data_test)
    X_test = [data_test[:n_test][0][:, :, 0], data_test[:n_test][0][:, :, 1:]]
    ypred = model.predict(X_test)
    ypred = np.argmax(ypred, axis=1)
    ytrue = np.argmax(data_test[:n_test][1], axis=1)

    acc = (ypred == ytrue).sum().item() / ytrue.shape[0]
    print("Acc: ", acc)

    class_acc = get_acc_per_class(ypred, ytrue)
    print("Class Acc: ", class_acc)

    mass_acc = get_acc_per_massbin(ypred, ytrue,
                                   data_test[:n_test][2])
    print("Mass-bin Acc: ", mass_acc)

    pT_acc = get_acc_per_pTbin(ypred, ytrue,
                               data_test[:n_test][3])
    print("pT-bin Acc: ", pT_acc)

    return acc, class_acc, mass_acc, pT_acc



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Nprong Model -- PFN')
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--tag", type=str)
    args = parser.parse_args()

    # -- output dit -- #
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # -- some tunable variables -- #
    device = 'cuda'
    lr = 1e-4  # or 1e-4
    epochs = 1000
    batch_size = 256
    #fold_id = None  # doing 10 bootstraps now

    fname = os.path.join(args.save_dir, "summary_{}.txt".format(args.tag))
    with open(fname, "w") as f:
        f.write("5-fold acc of the PFN.\n")
        f.write("batch size: {}\n".format(batch_size))
        f.write("learning rate: {}\n".format(lr))
        f.write("epochs: {}\n".format(epochs))
        f.write("Phi layers: [128 128]\n")
        f.write("F layers: [1024 1024]\n")
        f.write("F_dropouts: 0.2\n")
        f.write("Optimizer: Adam\n")
        f.write("Output act: Softmax")
        f.write("Hiden act: ReLu")
        f.write("Loss: CrossEntropyLoss\n")

        f.write("*****************************************\n")

        accuracy = []  # avg over all folds
        class_accuracy = []  # class avg over all folds
        mass_accuracy = []  # mass-bin avg over all folds
        pT_accuracy = []  # pT-bin avg over all folds
        for fold_id in range(1):
            print("*******************************************")
            print("* EFN Fold #: {}                           *".format(fold_id))
            print("*******************************************")
            # -- read and split the data -- #
            y, threeM, mass, pT = get_threeM_dataset(args.input_file)
            nsamples = y.shape[0]
            nclasses = len(np.unique(y))
            y = tf.keras.utils.to_categorical(y, num_classes=nclasses)
            print("Total of {} samples with {} classes".format(nsamples, nclasses))

            data_train, data_val, data_test = split_dataset(threeM, y, mass, pT,
                                                            fold_id=fold_id,
                                                            num_folds=10)
            # -- setup the model -- #
            model = setup_EFN(input_dim=2, lr=lr)
            def count_parameters(model):
                return np.sum([K.count_params(p) for p in set(model.trainable_weights)])
            f.write("Num. of trainable params: {} \n".format(count_parameters(model)))
            print("Total no. of params: ", count_parameters(model))

            # -- train & eval model -- #
            model_tag = "{}_{}".format(args.tag, fold_id)
            #train_model(model, args.save_dir, model_tag,
            #            data_train=data_train, data_val=data_val)

            acc, class_acc, mass_acc, pT_acc = eval_model(
                args.save_dir, model_tag, data_test=data_test
                )

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
        mean_class_accuracy = np.mean(class_accuracy, axis=0)
        mean_massbin_accuracy = np.mean(mass_accuracy, axis=0)
        mean_pTbin_accuracy = np.mean(pT_accuracy, axis=0)

        f.write("*****************************************\n")
        f.write("Avg accuracy: \n")
        f.write(str(mean_accuracy)+"\n")
        f.write("Avg class-bin accuracy: \n")
        f.write(str(mean_class_accuracy)+"\n")
        f.write("Avg mass-bin accuracy: \n")
        f.write(str(mean_massbin_accuracy)+"\n")
        f.write("Avg pT-bin accuracy: \n")
        f.write(str(mean_pTbin_accuracy)+"\n")
        f.write("*****************************************\n")

    print("Done :)")
