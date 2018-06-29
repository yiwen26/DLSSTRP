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

from keras.optimizers import SGD,Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV

from skimage.feature import blob_log

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

def get_model(learn_rate,dropoutP):

    '''
    get keras convolutional neural network model with given learning rate

    Arguments:learning rate, dropout rate

    Returns:keras model

    Raises:Error if the input is not a float.
    Error if dropout rate is larger than 1

    '''

    assert type(learn_rate)==float,"learning rate must be a float"
    assert dropoutP<1.,"drop out rate must less than 1"
    
    model = Sequential()

    ##### add layers to CNN model: 
    #model.add(Conv2D(8,kernel_size=(7,7),use_bias=False,strides=(1,1),
    #               input_shape=(64,64,1),kernel_initializer="glorot_normal"))
    #padding = 'same'
    model.add(Conv2D(8,kernel_size=(5,5),use_bias=False,strides=(1,1),
                     input_shape=(64,64,1),kernel_initializer="glorot_normal"))
    BatchNormalization(axis=1, momentum=0.99,epsilon=0.001,center=True)
    model.add(Activation("relu"))
    model.add(Conv2D(16,kernel_size=(5,5),use_bias=False,strides=(1,1),
                     kernel_initializer="glorot_normal"))
    BatchNormalization(axis=1, momentum=0.99,epsilon=0.001,center=True)
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Dropout(dropoutP))
    
    model.add(Conv2D(32,(5,5),use_bias=False,kernel_initializer="glorot_normal"))
    #model.add(Conv2D(16,(3,3),use_bias=False,kernel_initializer="glorot_normal"))
    BatchNormalization(axis=1, momentum=0.99,epsilon=0.001,center=True)
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Dropout(dropoutP))
    
    model.add(Conv2D(64,(3,3),use_bias=False,kernel_initializer="glorot_normal"))
    BatchNormalization(axis=1, momentum=0.99,epsilon=0.001,center=True)
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Dropout(dropoutP))
    
    model.add(Conv2D(128,(3,3),use_bias=False,kernel_initializer="glorot_normal"))
    BatchNormalization(axis=1, momentum=0.99,epsilon=0.001,center=True)
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Dropout(dropoutP))
    
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(dropoutP))
    model.add(Dense(6))
    BatchNormalization(axis=1, momentum=0.99,epsilon=0.001,center=True)
    model.add(Activation("softmax"))
    
    #sgd = SGD(lr=learn_rate,momentum=sgd_momentum)
    adam= Adam(lr=learn_rate)
    model.compile(loss = 'categorical_crossentropy',optimizer = adam, metrics=['accuracy'])
    return model



def gridsearch(x_train,y_train_origin,learn_rate,sgd_momentum,batch_size,epochs,filename):

    '''
    given the training dataset, it will be randomly split into training and validation set
    and the keras model will be trained with given learn_rate, batch size and epochs

    Arguments:
    x_train: x training dataset, need to be centered

    y_train_origin: y training dataset, needless to be centered

    learn_rate: a list of learning rate

    batchsize: a int

    epochs: a int

    filename: a folder will be created named by this filename

    Returns:
    fitresult: a list, each element is a keras history object
    models_grid: a list, each element is a trained model
    searchgrid: a list, the hyperparameters that have been searched

    every model, training history model_weights will be saved in a folder named learnrate

    Raises:
    error if the shape of x or y is wrong

    ''' 
    assert np.shape(x_train)[1:4]==(64, 64, 1),"the expected shape of x_train is (channels,img_x,img_y,1)"
    
    assert np.shape(y_train_origin)[1]==6,"the expected shape of y_train is (channels,6)"

    #create a new directory to save all the training history and model_weights
    version=1
    while os.path.exists('../HyperparametersTuning/'+filename+str(version)):
        version=version+1
    os.makedirs('../HyperparametersTuning/'+filename+str(version))
    
    search_grid=[(learn_rate[i],sgd_momentum[j]) 
                 for i in range(len(learn_rate)) for j in range(len(sgd_momentum))]

    fitresult=[] #list save history of each keras model
    models_grid=[get_model(learn_rate[0],sgd_momentum[0])]*len(search_grid) #list save each keras model
    
    for i in range(len(search_grid)):
        #randomly split the training data into training set and validation set
        x_tra,x_val,y_tra,y_val=train_test_split(x_train,y_train_origin,test_size=0.33)
        models_grid[i]=get_model(search_grid[i][0],search_grid[i][1])
        fitresult.append(models_grid[i].fit(x_tra,y_tra,
                                            batch_size=batch_size,
                                            epochs=epochs,
                                            verbose=1,
                                            validation_data=(x_val,y_val),shuffle=True))  
    
    with open('../HyperparametersTuning/'+filename+str(version)+"/search_grid.txt","wb") as fp:
        pickle.dump(search_grid,fp)
    
    for i in range(len(fitresult)):
        #save history of each model
        with open('../HyperparametersTuning/'+filename+str(version)+"/fitresult"+str(i)+".txt","wb") as fp:
            pickle.dump(fitresult[i].history,fp)
            
        #save each model and weights:
        models_grid[i].save('../HyperparametersTuning/'+filename+str(version)+'/model'+str(i)+'.h5')
        models_grid[i].save_weights('../HyperparametersTuning/'+filename+str(version)+'/model_weight'+str(i)+'.h5')
        
    return fitresult,models_grid,search_grid





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

def PlotLossAcc(load_res,search_grid):
    ''' 
    plot the acc, valacc, loss, val loss history in load_res
    argment: load_res: a list with all the training history
            search_grid: a list of hyperparameters that have been searched
    '''
    for j in range(len(load_res)):
        history=load_res[j]

        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5),sharex=True)
        ax1.plot(history['acc'])
        ax1.plot(history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        ax1.legend(['train', 'test'], loc='upper left')

        ax2.plot(history['loss'])
        ax2.plot(history['val_loss'])
        plt.title('(LearningRate,dropout) = '+str(search_grid[j]))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        ax2.legend(['train', 'test'], loc='upper left')
        plt.show()

    return


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


def visualize_class_activation_map(model_path, img_path):
    '''
    plot class activation map, and return the predicted defect type and coordination

    argments:model_path: keras CNN model
    img_path: original defect image, must be a numpy array
    '''

    model=model_path
    original_img=img_path
    width, height, _ = original_img.shape

    #Reshape to the network input shape (3, w, h).
    #img = np.array([np.transpose(np.float32(original_img), (0, 1,2))])
    img = np.array([np.transpose(np.float32(original_img), (0, 1,2))])
        
    #Get the 512 input weights to the softmax.
    class_weights = model.layers[-2].get_weights()[0]
    final_conv_layer = model.layers[15]   #because final convolutional layer is the 14th layer
    get_output = K.function([model.layers[0].input],[final_conv_layer.output, model.layers[-1].output])
    imglist=[]
    imglist.append(img)
    [conv_outputs, predictions] = get_output(imglist)
    conv_outputs = conv_outputs[0, :, :, :]

    #Create the class activation map.
    cam_origin = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])
    target_class = 4
    for i, w in enumerate(class_weights[:,target_class]):
        cam_origin += w * conv_outputs[:, :, i] # i th channel * w_i
        
    defects={0:'Sw',1:'Mo',2:'W2s2',3:'Vw',4:'Vs2',5:'Ws'}
    print("predicted defect type:")
    print(defects[np.where(predictions==np.max(predictions))[1][0]])
    
    cam_origin /= np.max(cam_origin)
    cam = cv2.resize(cam_origin, (height, width))
    heatmap=np.uint8(255*cam)
    heatmap[np.where(cam < 0.2)] = 0
    
    max_coord = blob_log(cam, min_sigma=0.8)
    x, y = np.transpose(max_coord)[0:2,:]
    coordinates = list(zip(x,y))
    print("coordinates: ")
    print(coordinates)
    
    original=255*(np.max(original_img)-original_img)/(np.max(original_img)-np.min(original_img))
    
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(np.reshape(original_img,(64,64)),cmap = 'gray')
    ax1.set_title('Original image')
    ax2.imshow(heatmap, cmap = 'jet')
    ax2.imshow(np.reshape(original,(64,64)), cmap = 'gray', alpha = 0.5)

    return heatmap
        


