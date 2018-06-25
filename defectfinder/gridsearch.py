from __future__ import print_function

from keras.layers import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from keras.models import Sequential

from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

import numpy as np
import pickle
import os.path


def detectGPU():
    """
    Detects if there is available GPU and show the devices list
    Arguments: None
    Returns: print available GPUs and Devices imformation
    Raises:None
    """

    from keras import backend as K
    from tensorflow.python.client import device_lib

    print("----Available GPUs(if GPU is detected, the code will automatically run on GPU)----")
    print(K.tensorflow_backend._get_available_gpus())
    print("----Devices imformation----")
    print(device_lib.list_local_devices())
    return


def get_model(learn_rate):
    """
    Get keras convolutional neural network model with given learning rate
    Arguments:learning rate
    Returns:keras model
    Raises:Error if the input is not a float
    """

    assert isinstance(learn_rate, float), "learning rate must be a float"

    model = Sequential()

    # add layers to CNN model:
    model.add(
        Conv2D(
            8, kernel_size=(
                7, 7), use_bias=False, strides=(
                1, 1), input_shape=(
                    64, 64, 1), kernel_initializer="glorot_normal"))
    # padding = 'same'
    BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True)
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(16, (5, 5), use_bias=False,
                     kernel_initializer="glorot_normal"))
    BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True)
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(32, (3, 3), use_bias=False,
                     kernel_initializer="glorot_normal"))
    BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True)
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(64, (3, 3), use_bias=False,
                     kernel_initializer="glorot_normal"))
    BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True)
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(6))
    BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True)
    model.add(Activation("softmax"))

    sgd = SGD(lr=learn_rate)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=sgd,
        metrics=['accuracy'])
    return model


def gridsearch(x_train, y_train_origin, learn_rate, batch_size, epochs):
    """
    Given the training dataset, it will be randomly split into training and
    validation setand the keras model will be trained with given learn_rate,
    batch size and epochs
    Arguments:
        x_train: x training dataset, need to be centered
        y_train_origin: y training dataset, needless to be centered
        learn_rate: a list of learning rate
        batchsize: an int
        epochs: an int
    Returns:
        fitresult: a list, each element is a keras history object
        models_grid: a list, each element is a trained model

        Every model, training history model_weights will be saved in a folder named learnrate
    Raises:error if the shape of x or y is wrong
    """

    assert np.shape(x_train)[1:4] == (
        64, 64, 1), "the expected shape of x_train is (channels,img_x,img_y,1)"

    assert np.shape(y_train_origin)[
        1] == 6, "the expected shape of y_train is (channels,6)"

    # create a new directory to save all the training history and model_weights
    version = 1
    while os.path.exists('./learn_rate' + str(version)):
        version = version + 1
    os.makedirs('./learn_rate' + str(version))

    fitresult = []  # list save history of each keras model
    models_grid = [get_model(learn_rate[0])] * \
        len(learn_rate)  # list save each keras model

    for i in range(len(learn_rate)):
        # randomly split the training data into training set and validation set
        x_tra, x_val, y_tra, y_val = train_test_split(
            x_train, y_train_origin, test_size=0.33)
        models_grid[i] = get_model(learn_rate[i])
        fitresult.append(
            models_grid[i].fit(
                x_tra,
                y_tra,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(
                    x_val,
                    y_val),
                shuffle=True))

    for i in range(len(fitresult)):
        # save history of each model
        with open('./learn_rate' + str(version) + "/fitresult" + str(i) + ".txt", "wb") as fp:
            pickle.dump(fitresult[i].history, fp)

        # save each model and weights:
        models_grid[i].save(
            './learn_rate' +
            str(version) +
            '/model' +
            str(i) +
            '.h5')
        models_grid[i].save_weights(
            './learn_rate' +
            str(version) +
            '/model_weight' +
            str(i) +
            '.h5')

    return fitresult, models_grid


def load_results(learn_rate, path):
    """
    Load training history for a list of model
    Arguments:
        learn_rate: learning rate
        path:the directory where you store all the training history and model_weights
    Returns: a list, each element is the training history of a model
    Raises: error if the path not exist
    """

    assert os.path.exists(path), "No such path"
    # path is the directory you save all the training history and model_weights

    load_res = [0] * len(learn_rate)
    for i in range(len(learn_rate)):
        with open(path + "/fitresult" + str(i) + ".txt", "rb") as fp:
            load_res[i] = pickle.load(fp)

    return load_res
