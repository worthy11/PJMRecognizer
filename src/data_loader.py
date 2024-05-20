import pandas as pd
import numpy as np
import glob
import os
import csv
from keras.utils import to_categorical

def LoadData(num_classes):
    sign_mnist_train = np.array(pd.read_csv("src/data/pjm_training_set.csv"))
    sign_mnist_test = np.array(pd.read_csv("src/data/pjm_testing_set.csv"))

    train_set = sign_mnist_train[:, 1:]
    train_labels = sign_mnist_train[:, 0]
    test_set = sign_mnist_test[:, 1:]
    test_labels = sign_mnist_test[:, 0]

    train_labels = to_categorical(train_labels, num_classes)
    test_labels = to_categorical(test_labels, num_classes)

    train_set = train_set.reshape(train_set.shape[0], 21, 21, 1)
    test_set = test_set.reshape(test_set.shape[0], 21, 21, 1)

    return (train_set, train_labels), (test_set, test_labels)

def findFiles(path): return glob.glob(path)

def LoadDataDynamic():
    # SAMPLE FORMAT: Python list with 22 rows: 
    # - row 0: absolute difference
    # - rows 1:21: relative difference

    train_set = []
    train_labels = []
    test_set = []
    test_labels = []

    with open('src/data/pjm_dynamic_training_set.csv') as file:
        reader = csv.reader(file, delimiter=',')
        sample = []
        for line in reader:
            if len(line) == 1:
                train_set.append(sample.copy())
                train_labels.append(int(line[0]))
                sample.clear()
            else:
                sample.append([float(_) for _ in line])

    with open('src/data/pjm_dynamic_testing_set.csv') as file:
        reader = csv.reader(file, delimiter=',')
        sample = []
        for line in reader:
            if len(line) == 1:
                test_set.append(sample)
                test_labels.append(int(line[0]))
                sample.clear()
            else:
                sample.append([float(_) for _ in line])

    # train_set = np.array(train_set)
    # train_labels = np.array(train_labels)
    # test_set = np.array(test_set)
    # test_labels = np.array(test_labels)

    # train_labels = to_categorical(train_labels, num_classes)
    # test_labels = to_categorical(test_labels, num_classes)

    # train_set = train_set.reshape(train_set.shape[0], 21, 21, 1)
    # test_set = test_set.reshape(test_set.shape[0], 21, 21, 1)
    
    return (train_set, train_labels), (test_set, test_labels)