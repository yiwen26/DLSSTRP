from __future__ import print_function

import keras
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.layers import Dense, Flatten, Dropout, Lambda, Activation
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D,BatchNormalization
from keras import layers
from keras.models import Sequential
from keras.models import load_model
from keras import backend as K
from keras.callbacks import *
from keras.initializers import glorot_normal
from keras.wrappers.scikit_learn import KerasClassifier

from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV

from IPython.display import clear_output
from matplotlib import pyplot as plt

import numpy as np

from scipy import fftpack
from scipy import ndimage

import cv2

import pickle
import os.path

def detectGPU():
    '''
    detect if there is available GPU and show the devices list

    Arguments: None

    Returns: print available GPUs and Devices imformation

    Raises:None

    '''
    
    from keras import backend as K
    from tensorflow.python.client import device_lib

    print("----Available GPUs(if GPU is detected, the code will automatically run on GPU)----")
    print(K.tensorflow_backend._get_available_gpus())
    print("----Devices imformation----")
    print(device_lib.list_local_devices())
    return


def get_model(learn_rate):

    '''
    get keras convolutional neural network model with given learning rate

    Arguments:learning rate

    Returns:keras model

    Raises:Error if the input is not a float.

    '''

    assert type(learn_rate)==float,"learning rate must be a float"
    
    model = Sequential()

    ##### add layers to CNN model: 
    model.add(Conv2D(8,kernel_size=(7,7),use_bias=False,strides=(1,1),
                     input_shape=(64,64,1),kernel_initializer="glorot_normal"))
    #padding = 'same'
    BatchNormalization(axis=1, momentum=0.99,epsilon=0.001,center=True)
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Conv2D(16,(5,5),use_bias=False,kernel_initializer="glorot_normal"))
    BatchNormalization(axis=1, momentum=0.99,epsilon=0.001,center=True)
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Conv2D(32,(3,3),use_bias=False,kernel_initializer="glorot_normal"))
    BatchNormalization(axis=1, momentum=0.99,epsilon=0.001,center=True)
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Conv2D(64,(3,3),use_bias=False,kernel_initializer="glorot_normal"))
    BatchNormalization(axis=1, momentum=0.99,epsilon=0.001,center=True)
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(GlobalAveragePooling2D())
    model.add(Dense(6))
    BatchNormalization(axis=1, momentum=0.99,epsilon=0.001,center=True)
    model.add(Activation("softmax"))
    
    sgd = SGD(lr=learn_rate)
    model.compile(loss = 'categorical_crossentropy',optimizer = sgd, metrics=['accuracy'])
    return model


def gridsearch(x_train,y_train_origin,learn_rate,batch_size,epochs):

    '''
    given the training dataset, it will be randomly split into training and validation set
    and the keras model will be trained with given learn_rate, batch size and epochs

    Arguments:
    x_train: x training dataset, need to be centered

    y_train_origin: y training dataset, needless to be centered

    learn_rate: a list of learning rate

    batchsize: a int

    epochs: a int

    Returns:
    fitresult: a list, each element is a keras history object
    models_grid: a list, each element is a trained model

    every model, training history model_weights will be saved in a folder named learnrate

    Raises:
    error if the shape of x or y is wrong

    ''' 
    assert np.shape(x_train)[1:4]==(64, 64, 1),"the expected shape of x_train is (channels,img_x,img_y,1)"
    
    assert np.shape(y_train_origin)[1]==6,"the expected shape of y_train is (channels,6)"

    #create a new directory to save all the training history and model_weights
    version=1
    while os.path.exists('../HyperparametersTuning/learn_rate'+str(version)):
        version=version+1
    os.makedirs('../HyperparametersTuning/learn_rate'+str(version))

    fitresult=[] #list save history of each keras model
    models_grid=[get_model(learn_rate[0])]*len(learn_rate) #list save each keras model

    for i in range(len(learn_rate)):
        #randomly split the training data into training set and validation set
        x_tra,x_val,y_tra,y_val=train_test_split(x_train,y_train_origin,test_size=0.33)
        models_grid[i]=get_model(learn_rate[i])
        fitresult.append(models_grid[i].fit(x_tra,y_tra,
                                            batch_size=batch_size,
                                            epochs=epochs,
                                            verbose=1,
                                            validation_data=(x_val,y_val),shuffle=True))  
    
    
    for i in range(len(fitresult)):
        #save history of each model
        with open('../HyperparametersTuning/learn_rate'+str(version)+"/fitresult"+str(i)+".txt","wb") as fp:
            pickle.dump(fitresult[i].history,fp)
    
        #save each model and weights:
        models_grid[i].save('../HyperparametersTuning/learn_rate'+str(version)+'/model'+str(i)+'.h5')
        models_grid[i].save_weights('../HyperparametersTuning/learn_rate'+str(version)+'/model_weight'+str(i)+'.h5')
        
    return fitresult,models_grid



def load_results(learn_rate,path):
    '''
    Load training history for a list of model

    Arguments:
    learning rate
    path:the directory where you store all the training history and model_weights

    Returns: a list, each element is the training history of a model

    Raises: error if the path not exist

    '''
    
    assert os.path.exists(path), "No such path"
    #path is the directory you save all the training history and model_weights
    
    load_res=[0]*len(learn_rate)
    for i in range(len(learn_rate)):
        with open(path+"/fitresult"+str(i)+".txt","rb") as fp:
            load_res[i]=pickle.load(fp)
            
    return load_res


class PlotLearning(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5),sharex=True)
        
        clear_output(wait=True)
        
        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.legend()
        
        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend()
        
        plt.show();
        


