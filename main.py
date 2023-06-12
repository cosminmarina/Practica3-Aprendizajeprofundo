from __future__ import annotations
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, roc_auc_score, f1_score, recall_score, precision_recall_fscore_support, ConfusionMatrixDisplay
import ArchModels
import tensorflow as tf
import keras
from keras.utils import to_categorical
from keras import layers
from typing import Union
import glob
from PIL import Image
import argparse
import warnings
import json
import keras_tuner

def load_data(path:str, size:tuple=(100,100), test_size:float=0.25, verbose:bool=True):
    # Extract files names
    a_type = glob.glob(f"{path}/Tipo A: Kunzea/*.png")
    b_type = glob.glob(f"{path}/Tipo B: Lepto/*.png")

    if verbose:
        print(f"Amount of A type: {np.shape(a_type)}")
        print(f"Amount of B type: {np.shape(b_type)}")
        print(len(size))

    # Load images
    if size == (100,100):
        a_imgs = np.array([np.array(Image.open((img)).convert('L')) for img in a_type])
        b_imgs = np.array([np.array(Image.open((img)).convert('L')) for img in b_type])
    elif len(size) == 3:
        a_imgs = np.array([np.array(Image.open((img)).convert('L').resize(size[:2])) for img in a_type])
        b_imgs = np.array([np.array(Image.open((img)).convert('L').resize(size[:2])) for img in b_type])
        a_imgs = np.reshape([a_imgs], (np.shape(a_imgs) + (1,)))
        b_imgs = np.reshape([b_imgs], (np.shape(b_imgs) + (1,)))
    else:
        a_imgs = np.array([np.array(Image.open((img)).convert('L').resize(size)) for img in a_type])
        b_imgs = np.array([np.array(Image.open((img)).convert('L').resize(size)) for img in b_type])

    if verbose:
        print(f"A type shape: {np.shape(a_imgs)}")
        print(f"B type shape: {np.shape(b_imgs)}")

    # Make dataset
    y = np.concatenate((np.zeros(np.shape(a_imgs)[0]), np.ones(np.shape(b_type)[0])), axis=0)
    y = to_categorical(y)
    X = np.concatenate((a_imgs, b_imgs), axis=0)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test

def get_stats(y_test: Union[np.ndarray, list, tf.data.DataSet], y_pred: Union[np.ndarray, list, tf.data.DataSet], file_save:str, arch:int=3):
    """
    | get_stats                                         |
    |---------------------------------------------------|
    | Function that obtain the scores of the classifier.|
    |___________________________________________________|
    | ndarray, ndarray, str ->                          |
    |___________________________________________________|
    | Input:                                            |
    | y_test, y_pred: the real and predicted y values.  |
    | file_save: where to save the results.             |
    | latent_dim: int that represent the shape of the   |
    |    latent (code) space.                           |
    |___________________________________________________|
    | Output:                                           |
    | Nothing, we save all in files.                    |
    """
    # RECALL
    rec_gen = recall_score(y_test, y_pred, average='weighted')
    # PRECISION
    prec_gen = precision_score(y_test, y_pred, average='weighted')
    # F1
    f1_gen = f1_score(y_test, y_pred, average='weighted')
    # PER CLASS
    prf = precision_recall_fscore_support(y_test, y_pred, average=None)
    
    gen_stats = np.array([prec_gen, rec_gen, f1_gen])
    df = pd.DataFrame(np.concatenate((prf[:-1], np.array([gen_stats]).T), axis=1), index=['Precision', 'Recall', 'Fscore'], columns=['Tipo A: Kunzea', 'Tipo B: Lepto', 'General'])
    df.to_csv(f'./stats-{file_save[2:-3]}-arch{arch}.csv')

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.savefig(f'./confusion_matrix-{file_save[2:-3]}-arch{arch}.png')
    return

def runModel(input_dim:Union[np.ndarray, list], arch:int, with_cpu:bool, n_epochs:int, data_train:Union[np.ndarray, list, tf.data.DataSet], data_test:Union[np.ndarray, list, tf.data.DataSet], target_train:Union[np.ndarray, list, tf.data.DataSet], target_test:Union[np.ndarray, list, tf.data.DataSet], file_save:str, verbose:bool=True):
    verbose = 1 if verbose else 0
    if with_cpu:
        with tf.device("/cpu:0"):
            am = ArchModels.ArchModels(input_dim=input_dim, arch=arch)

            am.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            history = am.fit(data_train, target_train, epochs=n_epochs, batch_size=32, min_delta=1e-2, patience=40, verbose=verbose)
    else:
        am = ArchModels.ArchModels(input_dim=input_dim, arch=arch)

        am.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        history = am.fit(data_train, target_train, epochs=n_epochs, batch_size=32, min_delta=1e-2, patience=40, verbose=verbose)
    
    plt.figure()
    plt.plot(history.history['loss'],label='loss')
    plt.plot(history.history['val_loss'],label='val_loss')
    plt.legend()
    plt.savefig(f'./history-loss-{file_save[2:-3]}-arch{arch}.png')
    plt.yscale('log')
    plt.savefig(f'./history-loss-{file_save[2:-3]}-arch{arch}-logscale.png')

    plt.figure()
    plt.plot(history.history['accuracy'],label='accuracy')
    plt.plot(history.history['val_accuracy'],label='val_accuracy')
    plt.legend()
    plt.savefig(f'./history-accuracy-{file_save[2:-3]}-arch{arch}.png')
    plt.yscale('log')
    plt.savefig(f'./history-accuracy-{file_save[2:-3]}-arch{arch}-logscale.png')

    am.model.save(file_save)
    y_pred = am.model.predict(data_test)
    get_stats(np.argmax(target_test, axis=1), np.argmax(y_pred, axis=1), file_save, arch)
    return history


def keras_code(X, y, params, filter1, filter2, kernel1, kernel2, activation1, activation2, optimizer, file_save):
    am = ArchModels.ArchModels(params["input_dim"], 3, 1, 2, filter1, filter2, kernel1, kernel2, activation1, activation2)
    am.model.compile(optimizer, loss='categorical_crossentropy')
    X_train, X_val, y_train, y_val = train_test_split(X, y)
    history = am.model.fit(X_train, y_train)
    am.model.save(file_save)
    y_pred = am.model.predict(X_val)
    return np.mean(np.abs(y_pred - y_val))

class MyTuner(keras_tuner.RandomSearch):
    def __init__(self, X, y, params, **kwargs):
        keras_tuner.RandomSearch.__init__(self)
        self.X = X
        self.y = y
        self.params = params

    def run_trial(self, trial, **kwargs):
        hp = trial.hyperparameters
        return keras_code(self.X, self.y, self.params,
            filter1=hp.Int("filter1", 16, 256, 16),
            filter2=hp.Int("filter2", 16, 256, 16),
            kernel1=hp.Int("kernel1", 1, 4, 1),
            kernel2=hp.Int("kernel2", 1, 4, 1),
            activation1=hp.Choice("activation1", ["softmax", "selu", "sigmoid"]),
            activation2=hp.Choice("activation2", ["softmax", "selu", "sigmoid"]),
            optimizer=hp.Choice("optimizer", ["adam", "adadelta", "ftrl"]),
            file_save=os.path.join("./tmp", trial.trial_id),
        )

def tun_net3(X, y, params):
    tuner = MyTuner(
        X=X, y=y, params=params,
        max_trials=3, overwrite=True, directory="mydir", project_name="optimize_convolution",
    )
    tuner.search()
    # Retraining the model
    best_hp = tuner.get_best_hyperparameters()[0]
    print(best_hp.values)
    with open(params["save_best"], 'w') as f:
        json.dump(best_hp.values, f)
    keras_code(X, y, params, **best_hp.values, file_save="./tmp/best_model")

def main():
    file_params_name = 'params.json'
    
    # Parser initialization
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", dest='method', help="Specify an method to execute between: \n 'all' (default), 'net' or 'tun'")
    parser.add_argument("-f", "--configfile", dest='conf', help="JSON file with configuration of parameters. If not specified and 'method' require the file, it will be searched at 'params.json'")
    args = parser.parse_args()
    
    # Read file
    ## Default if configfile is not specified
    if args.conf is not None:
        file_params_name = args.conf
    else:
        warnings.warn("Not specified configuration file (-f | --configfile), by default it will be searched at 'params.json'")
    ## Try-except-else to open file and re-write params
    try:
        file_params = open(file_params_name)
    except:
        raise OSError(f'File {file_params_name} not found. Your method need a configuration parameters file')
    else:
        params = json.load(file_params)
        file_params.close()
    # All nets
    if args.method == 'all' or args.method is None:
        X_train, X_test, y_train, y_test = load_data('./anuka1200/', size=tuple(params["input_dim"]), verbose=True)
        if params["verbose"]:
            print('X Train: ', np.shape(X_train))
            print('X Test: ', np.shape(X_test))
            print('y Train: ', np.shape(y_train))
            print('y Test: ', np.shape(y_test))

        list_history = []
        for i in range(5):
            history = runModel(input_dim=params["input_dim"], arch=i+1, with_cpu=params["with_cpu"] ,n_epochs=params["n_epochs"], data_train=X_train, data_test=X_test, target_train=y_train, target_test=y_test, file_save=params["file_save"], verbose=params["verbose"])
            list_history.append(100*np.array(history.history['val_accuracy']))
        plt.figure()
        plt.plot(np.arange(len(list_history[0])), list_history[0], label='Accuracy Net1')
        plt.plot(np.arange(len(list_history[1])), list_history[1], label='Accuracy Net2')
        plt.plot(np.arange(len(list_history[2])), list_history[2], label='Accuracy Net3')
        plt.plot(np.arange(len(list_history[3])), list_history[3], label='Accuracy Net4')
        plt.plot(np.arange(len(list_history[4])), list_history[4], label='Accuracy Net5')
        plt.legend()
        plt.savefig(f'./history-accuracy-{params["file_save"][2:-3]}-all.png')
        plt.yscale('symlog')
        plt.savefig(f'./history-accuracy-{params["file_save"][2:-3]}-all-logscale.png')

    # 1 net
    elif args.method == 'net':
        X_train, X_test, y_train, y_test = load_data('./anuka1200/', size=tuple(params["input_dim"]), verbose=True)
        if params["verbose"]:
            print('X Train: ', np.shape(X_train))
            print('X Test: ', np.shape(X_test))
            print('y Train: ', np.shape(y_train))
            print('y Test: ', np.shape(y_test))
        runModel(input_dim=params["input_dim"], arch=params["arch"], with_cpu=params["with_cpu"], n_epochs=params["n_epochs"], data_train=X_train, data_test=X_test, target_train=y_train, target_test=y_test, file_save=params["file_save"], verbose=params["verbose"])
    # Tunning
    elif args.method == 'tun':
        X_train, X_test, y_train, y_test = load_data('./anuka1200/', size=tuple(params["input_dim"]), verbose=True)
        if params["verbose"]:
            print('X Train: ', np.shape(X_train))
            print('X Test: ', np.shape(X_test))
            print('y Train: ', np.shape(y_train))
            print('y Test: ', np.shape(y_test))
        tun_net3(X_train, y_train, params)
    else:
        raise ValueError(f"Not recognized {args.method} method. The availabre methods are 'all' (default), 'net' or 'tun'.")


if __name__ == "__main__":
    main()