import numpy as np
import keras
from keras import backend as K


def image_preprocessing_cam(image_data, image_size):
    
    """
    Preprocessing for class activation maps. The function will reshape the image data and normalize it.
    
    Arguments:    
    image_data : this is the image that is read in
    image_size : this is the size of the image that is required for the model to analyse
    
    Returns:
    Normalized and reshaped image array
    
    Raises:
    Error if the input is incorrect
    """
    
    assert type(image_size) == tuple, ('Wrong data type', 'Must be a tuple')
    assert type(image_data) == np.ndarray, ('Wrong data type', 'Must be a numpy array')
    
    image_data = image_data.reshape(1, image_size[0], image_size[1], 1)
    image_data = image_data.astype('float32')
    image_data = (image_data - np.amin(image_data))/(np.amax(image_data) - np.amin(image_data))
    
    return image_data



def get_predictions(model, image):
    
    """
    Predicts the target class in the image provided using the model provided
    
    Arguments:
    model : This is the CNN model from where you want to extract the softmax weights
    image : This is the image where you want to predict the defect
    
    Returns:
    Predictions and the target class of the image (or the defect type in the image)
    
    Raises:
    Error if the input data type is incorrect
    """
    
    assert type(model) == keras.models.Sequential, ('Wrong data type', 'It should be a sequential keras model')
    assert type(image) == np.ndarray, ('Wrong data type', 'Must be a numpy array')

    predictions = model.predict(image)
    target_class = predictions.argmax()
    
    return predictions, target_class



def get_activation_maps(model, inputs, layer_num, learning_phase=0):
    
    
    get_layer_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer_num].output])
    layer_out = get_layer_output([inputs, learning_phase])[0]
    layer_out = layer_out[0, :, :, :]
    layer_out = np.transpose(layer_out,(2, 0, 1))
    return layer_out



def get_SoftmaxWeights(model):
    
    """
    This function provides the softmax weights from the model.
    
    Arguments:
    model : This is the CNN model from where you want to extract the softmax weights
    
    Returns:
    Softmax weights from the model
    
    Raises:    
    Error when the input data type is wrong
    """
    assert type(model) == keras.models.Sequential, ('Wrong data type', 'It should be a sequential keras model' )
    
    return model.layers[-1].get_weights()[0]



def get_defects(defects, target_class):
    
    """
    Provides the exact type of defect by using target class predicted and defect dictinary provided.
    
    Arguments:
    defects : This is the defect dictionary contianing all possible types of defects
    target_class : This is the target class predicted using the 'get_predictions' functions
    
    Returns:
    Defect type in the image provided
    
    Raises:
    Error if the input type is incorrect
    """
    
    assert type(defects) == dict, ('Wrong data type', 'Must be a dictionary')
    #assert type(target_class) == np.int64, ('Wrong data type', 'Must be an integer')
    
    for i, key in enumerate(defects.keys()):
        if i == target_class:
            return key
