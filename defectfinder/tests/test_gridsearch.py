'''
You may receive RuntimeError: Python is not installed as a framework
To solve this problem, please create a file: 
~/.matplotlib/matplotlibrc
and add the code: backend: TkAgg

'''

from __future__ import absolute_import, division, print_function

import keras
import sys
import numpy as np

sys.path.append('..')
import gridsearch

def test_get_model():

    '''
    test if get_model() return a sequential model
    test if there will be an error when input a list of learning rate

    '''
    model=gridsearch.get_model(0.5)
    assert isinstance(model,keras.engine.sequential.Sequential),"Not return a keras sequential model"

    try:
        model=gridsearch.get_model([0.1,0.05,0.001])
    except (AssertionError):
        pass
    else:
        raise Exception('Readin a list! The input should be a fload!')
    
    return


def test_gridsearch():
    '''
    test if there will be an error when input wrong shape of data

    '''
    
    x_train=np.random.rand(50,64,64,1)
    y_train=np.random.randint(6,size=(50,5))
    
    try:
        result,models=gridsearch.gridsearch(x_train,y_train,[0.01,0.02],128,20)
    except (AssertionError):
        pass
    else:
        raise Exception('Input wrong shape')
        
    return


def test_load_results():
    '''
    test if load_results() returns a list
    test if it returns the history of validation loss,validation accuracy, loss and accuracy
    
    '''
    
    learn_rate=(10**np.random.uniform(-5,0,15)).tolist()
    load_res=gridsearch.load_results(learn_rate,"../../HyperparametersTuning/learn_rate")
    
    assert isinstance(load_res,list),"Not return a list"
    
    for i in range(len(load_res)):
        assert list(load_res[i].keys())==['val_loss', 'val_acc', 'loss', 'acc'],"Missing loss or accuracy history"
    
    try:
        gridsearch.load_results(learn_rate,"../HyperparametersTuning/learn_rate")
    except(AssertionError):
        pass
    else:
        raise Exception('Input wrong path')
    
    return
    


        
