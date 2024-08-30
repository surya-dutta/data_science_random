# import common libraries and set fixed values
import numpy as np
import pandas as pd
import pdb as pdb
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt

# SciPy libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

# import PyTorch libraries
import torch
import torch.nn as nn

# import Keras libraries
from keras.layers import Dense, Input, ReLU
from keras.activations import sigmoid
from keras.models import Sequential, Model
from keras import regularizers
from keras.losses import binary_crossentropy
from keras import backend as K
import Balance
#execution_path = "../../../"

import importlib.util
# ROOT_DIR = '/content/gdrive/MyDrive/ResearchProject/BalancingAlgorithms'
# ROOT_DIR = Path(f"{execution_path}Common_Utils/BalancingAlgorithms/")

spec = importlib.util.spec_from_file_location('balance', 'Balance.py')

balance_method = importlib.util.module_from_spec(spec)
spec.loader.exec_module(balance_method)

def split_data(target, data, class_var,test_frac=0.2, transfer=False, balance=False):
    # to hold dataset
    dataset = pd.DataFrame()
    if transfer == True:
        dataset = pd.concat([data[key].copy() for key in data.keys() if key != target])
    else:
        dataset = data[target].copy()
    X = dataset.drop([class_var], axis=1)
    #X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns.values)
    Y = dataset.loc[:,class_var:class_var] #what is going on here? Why not just loc['class_var']
    #pdb.set_trace();
    if transfer == True:
        X_train, X_test, Y_train, Y_test = train_test_split(X.to_numpy(), Y.to_numpy(), test_size=test_frac, shuffle=True)

        if balance == True:
            # create train dataset as DataFrame
            df_X_train = pd.DataFrame(X_train, columns=X.columns.values)
            df_Y_train = pd.DataFrame(Y_train, columns=Y.columns.values)
            train_dataset = df_X_train.join(df_Y_train)

            # train-upsampling (random copy of minority class)
            train_dataset = balance_method.ADKNN(train_dataset, classIndex=class_var, minorityLabel=0, printDebug=False)
            # convert train dataset into X_train and Y_train
            X_train = train_dataset.drop(class_var, axis=1).to_numpy()
            Y_train = train_dataset.loc[:,class_var:class_var].to_numpy()
        return X_train, X_test, Y_train, Y_test

    else:
        dataset_list = []
        kf = KFold(n_splits=3, shuffle=True)
        for train_index, test_index in kf.split(X.to_numpy()):
            X_train, X_test = X.to_numpy()[train_index], X.to_numpy()[test_index]
            Y_train, Y_test = Y.to_numpy()[train_index], Y.to_numpy()[test_index]

            if balance == True:
                # create train dataset as DataFrame
                df_X_train = pd.DataFrame(X_train, columns=X.columns.values)
                df_Y_train = pd.DataFrame(Y_train, columns=Y.columns.values)
                train_dataset = df_X_train.join(df_Y_train)

                # train-upsampling (random copy of minority class)
                #train_dataset = balance_method.ADKNN(train_dataset, classIndex=class_var, minorityLabel=0, printDebug=False)
                train_dataset = Balance.generate_autoencoder(train_dataset, class_var=class_var, minority_var=0, printDebug=False)
            
                
                
                
                # convert train dataset into X_train and Y_train
                X_train = train_dataset.drop(class_var, axis=1).to_numpy()
                Y_train = train_dataset.loc[:,class_var:class_var].to_numpy()
            dataset_list.append([X_train, X_test, Y_train, Y_test])
        return dataset_list

def evaluate(y_hat_class, Y): # Y = real value, Y_hat = expected
    cm = np.array([[0, 0], [0, 0]])
    if all(y_hat_class == Y.reshape(-1,1)):
        cm[0][0] += sum(y_hat_class == 0)
        cm[1][1] += sum(y_hat_class == 1)
    else:
        cm = confusion_matrix(Y.reshape(-1, 1), y_hat_class)

    # print("cm",cm)
    # print("cm.ravel()",cm.ravel())
    tn, fp, fn, tp = cm.ravel()
    assert (tp + tn + fp + fn) != 0.0
    # a = (tp + tn) / (tp + tn + fp + fn)                                       ## Removed - Accuracy
    wa0 = (tn / (2 * (tn + fp))) if (tn + fp) != 0.0 else 0.0                   ## Local Var - Weighted_Accuracy_class0
    wa1 = (tp / (2 * (tp + fn))) if (tp + fn) != 0.0 else 0.0                   ## Local Var - Weighted_Accuracy_class1
    wa = wa0 + wa1                                                              ## Weighted Accuracy
    s = tp / (tp + fn) if (tp + fn) != 0.0 else 0.0                             ## Specificity
    p0 = tn / (tn + fn) if (tn + fn) != 0.0 else 0.0                            ## Precision_class0
    p1 = tp / (tp + fp) if (tp + fp) != 0.0 else 0.0                            ## Precision_class1
    r = tn / (tn + fp) if (tn + fp) != 0.0 else 0.0                             ## Sensitivity/Recall
    fscore0 = 2 * p0 * r / (p0 + r) if p0 + r != 0.0 else 0.0                   ## F1_class0
    fscore1 = 2 * p1 * s / (p1 + s) if p1 + s != 0.0 else 0.0                   ## F1_class1
    # d = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) != 0.0 else 0.0     ## Removed - Accuracy
    j = tn / (tn + fp + fn) if (tn + fp + fn) != 0.0 else 0.0                   ## Jaccard
    tpr = tp / (fn + tp) if (fn + tp) != 0 else 0.0                             ## Local Var - True Positive Rate
    fpr = fp / (tn + fp) if (tn + fp) != 0 else 0.0                             ## False Positive Rate
    tnr = tn / (tn + fp) if (tn + fp) != 0 else 0.0                             ## Local Var - True Negative Rate
    fnr = fn / (fn + tp) if (fn + tp) != 0 else 0.0                             ## False Negative Rate
    fdr = fp / (tp + fp) if (tp + fp) != 0 else 0.0                             ## False Discovery Rate
    fo_rate = fn / (fn + tn) if (fn + tn) != 0 else 0.0                         ## False Omission Rate
    # Formula for AUC_Roc score without using function - https://stackoverflow.com/questions/50848163/manually-calculate-auc
    auc_roc = (1/2) - (fpr/2) + (tpr/2)                                         ## auc_roc score
    pavg = (p0 + p1) / 2.0                                                      ## Precision_avg
    f1avg = (fscore0 + fscore1) / 2.0                                           ## F1_avg
    return np.array([wa, r, s, p0, p1, pavg, fscore0, fscore1, f1avg, auc_roc, fpr, fdr, fnr, fo_rate, j]), cm

# def predict(X):
#     return nn_ind.predict(X)

def test(X_test, Y_test):
    y_hat = predict(X_test)
    y_hat_class = (y_hat+0.5).astype(int)
    return evaluate(y_hat_class, Y_test)

def save_confusion_matrix(cms, filename):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20,5))
    for i in range(len(cms)):
        cm = cms[i]
        cax = axs[i].matshow(cm, cmap='binary', interpolation='nearest')
        axs[i].set_title(f'Confusion Matrix Round #{i + 1}', fontsize='xx-large')
        axs[i].set_xlabel('Predicted', fontsize='xx-large')
        axs[i].set_ylabel('True', fontsize='xx-large')
        axs[i].set_xticks([0, 1])
        axs[i].set_xticklabels(['0', '1'], fontsize='xx-large')
        axs[i].set_yticks([0, 1])
        axs[i].set_yticklabels(['0', '1'], fontsize='xx-large')
        axs[i].xaxis.set_ticks_position('bottom')
        for (ii, jj), z in np.ndenumerate(cm):
            axs[i].text(jj, ii, '{:0.1f}'.format(z),
                        bbox=dict(facecolor='white', edgecolor='0.3'),
                        ha='center', va='center', fontsize='xx-large')
    plt.savefig(filename+'.png', bbox_inches='tight')
    plt.close()