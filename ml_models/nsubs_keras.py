import os
import sys
import argparse
import time
import h5py
import numpy as np
from utils import plot_loss_acc
from utils import get_acc_per_class, get_acc_per_massbin, get_acc_per_pTbin

import tensorflow as tf
from tensorflow import keras

def get_nsub_dataset(input_file):
    with h5py.File(input_file, 'r') as f:
        y = np.array(f['target'])
        nsubs = np.concatenate((f['Nsubs']['Nsubs_beta05'],
                                f['Nsubs']['Nsubs_beta10'],
                                f['Nsubs']['Nsubs_beta20']), axis=-1)
        #mass = np.array(f['jet_Mass'])
        #pT = np.array(f['jet_PT'])
    #mass_pT_file = "/home/alex/Desktop/Nprong_AR/datasets/dataset_noPtNorm.h5"
    mass_pT_file = "/Users/alex/Desktop/Nprong/ml_models/dataset/dataset_noPtNorm.h5"
    with h5py.File(mass_pT_file, 'r') as f:
        mass = np.array(f['jet_Mass'])
        pT = np.array(f['jet_PT'])
    return y, nsubs, mass, pT

def cross_validate_perm(data, fold_id, num_folds=10):
    n = data.shape[0]
    test_size = int(n // num_folds)
    test_ids = (fold_id * test_size + np.arange(test_size)).tolist()
    perm = np.array([i for i in range(n) if i not in test_ids] + test_ids)
    return data[perm]

def split_nsub_dataset_temp(nsubs, y, mass, pT, fold_id=None, num_folds=10):
    total_num_sample = nsubs.shape[0]
    if fold_id is not None:
        l = [nsubs, y, mass, pT]
        # Rearrange the samples. The test samples are placed at the end.
        for i in range(len(l)):
            l[i] = cross_validate_perm(l[i], fold_id=fold_id,
                                       num_folds=num_folds)
        nsubs, y, mass, pT = l

    # for 10-folds, the split train:test:val is 80:10:10
    train_cut = int(total_num_sample * 0.8)
    val_cut = int(total_num_sample * 0.9)

    nsubs_train, nsubs_val, nsubs_test = \
        nsubs[:train_cut], nsubs[train_cut:val_cut], nsubs[val_cut:]
    y_train, y_val, y_test = \
        y[:train_cut], y[train_cut:val_cut], y[val_cut:]
    mass_train, mass_val, mass_test = \
        mass[:train_cut], mass[train_cut:val_cut], mass[val_cut:]
    pT_train, pT_val, pT_test = \
        pT[:train_cut], pT[train_cut:val_cut], pT[val_cut:]

    # scaler = StandardSaler()
    # nsubs_train = scaler.fit_transform(nsubs_train)
    # nsubs_val = scaler.transform(nsubs_val)
    # nsubs_test = scaler.transform(nsubs_test)

    data_train = (nsubs_train, y_train, mass_train, pT_train)
    data_val = (nsubs_val, y_val, mass_val, pT_val)
    data_test = (nsubs_test, y_test, mass_test, pT_test)
    return data_train, data_val, data_test

def HLNet():
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(135,)))
    model.add(tf.keras.layers.Dense(800, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(800, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(800, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(800, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(800, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(800, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(7, activation="softmax"))
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=opt,
                  metrics=["accuracy"],
                  loss="categorical_crossentropy")
    return model



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Nprong Model')
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--tag", type=str)
    args = parser.parse_args()

    # -- output dit -- #
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # -- some tunable variables -- #
    device = 'cuda'
    lr = 1e-3  # or 1e-4
    epochs = 10
    batch_size = 256
    #fold_id = None  # doing 10 bootstraps now

    # -- read and split the data -- #
    y, nsubs, mass, pT = get_nsub_dataset(args.input_file)
    nsamples = y.shape[0]
    nclasses = len(np.unique(y))
    print("Total of {} samples with {} distinct classes".format(nsamples,
                                                                nclasses))

    f = open(
        os.path.join(args.save_dir, "summary_{}.txt".format(args.tag)), "w")
    f.write("10-fold acc of the FNN trained on 135 N-sub variables.\n")
    f.write("batch size: {}\n".format(batch_size))
    f.write("learning rate: {}\n".format(lr))
    f.write("epochs: {}\n".format(epochs))
    f.write("Net layers: [135-800-800-800-800-800-64-7]\n")
    f.write("Dropout: 0.3\n")
    f.write("Hidden act: ReLU (up until 64-u layer)\n")
    f.write("Last two layers act: None (why?)\n")
    f.write("Optimizer: Adam\n")
    f.write("Loss: CrossEntropyLoss\n")

    f.write("*****************************************\n")


    accuracy = []  # avg over all folds
    class_accuracy = []  # class avg over all folds
    mass_accuracy = []  # mass-bin avg over all folds
    pT_accuracy = []  # pT-bin avg over all folds
    for fold_id in range(10):
        data_train, data_val, data_test = split_nsub_dataset_temp(
            nsubs, y, mass, pT, fold_id=fold_id, num_folds=10)

        (nsubs_train, y_train, mass_train, pT_train) = data_train
        (nsubs_val, y_val, mass_val, pT_val) = data_val
        (nsubs_test, y_test, mass_test, pT_test) = data_test
        # -- setup the model -- #
        model = HLNet()
        print(model.summary())

        model_tag = "{}_{}".format(args.tag, fold_id)
        saved_best_model = os.path.join(args.save_dir,
                                        "best_model_{}.pt".format(model_tag))
        callbacks = tf.keras.callbacks.ModelCheckpoint(saved_best_model, monitor="val_acc",)
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=7)
        y_val = tf.keras.utils.to_categorical(y_val, num_classes=7)
        history = model.fit(x=nsubs_train, y=y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(nsubs_val, y_val),
                            callbacks=callbacks,
                            workers=8,
                            use_multiprocessing=True)

        ypred = model.predict(nsubs_test)
        ypred = np.argmax(ypred, axis=1)

        class_acc = get_acc_per_class(ypred, y_test)
        print("Class Acc: ", class_acc)

        mass_acc = get_acc_per_massbin(ypred, y_test, mass_test)
        print("Mass-bin Acc: ", mass_acc)

        pT_acc = get_acc_per_pTbin(ypred, y_test, pT_test)
        print("pT-bin Acc: ", pT_acc)

        plot_loss_acc(history.history['loss'],
                      history.history['accuracy'],
                      history.history['val_loss'],
                      history.history['val_accuracy'],
                      args.save_dir, model_tag)

        acc, class_acc, mass_acc, pT_acc = main(model, args.save_dir, model_tag)
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
    f.close()

    print("Done :)")
