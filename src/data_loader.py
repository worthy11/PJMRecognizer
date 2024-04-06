import pandas as pd
import numpy as np
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