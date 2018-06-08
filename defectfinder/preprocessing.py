import os
from skimage.io import imsave
from skimage.color import rgb2gray
from skimage.util import random_noise
from keras.preprocessing.image import ImageDataGenerator, img_to_array


"""---------------------------------Function for to image augmentation-------------------------------"""

def AugmentImage(img, path, folder_name, image_prefix, N):
    
    """ This function uses the keras ImageDataGenerator for image augmentation of images 
    stored in the folder accessed through the defined "path".
    
    Parameters used for image augmentation :
    Rotation range of 10 degress (to preserve lattice symmtery)
    Shear range of 0.2 
    Zoom range of 70% to 100% 
    Vertical and horizontal flips are also included
    
    Arguments:
    
    img : Image that needs to be augmented
    path : Path to the raw images that need to be augmented
    folder_name : name of the folder where the images will be saved
    image_prefix : prefix for the saved images
    N : number of images that will be created through augmentation
    
    Returns:
    
    N augmented images with 'image_prefix' saved in 'folder_name'
    
    Raises:
    
    Error when input data type is incorrect
    
    """
    
    assert type(path) == str, ('Wrong Data type', 'path must be a string')
    assert type(folder_name) == str, ('Wrong Data type', 'folder_name must be a string')
    assert type(image_prefix) == str, ('Wrong Data type', 'image_prefix must be a string')
    assert type(N) == int, ('Wrong Data type', 'N must be an integer')
    
    directory = path +"/"+image_prefix+"/"+folder_name
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    datagen = ImageDataGenerator(rotation_range=10,
                                 shear_range=0.2,
                                 zoom_range=[0.7,1],
                                 horizontal_flip=True, 
                                 vertical_flip=True, 
                                 fill_mode='constant', 
                                 cval = 0.0)

    
    x = img_to_array(img)  # this is a Numpy array with shape (64, 64, 1)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 64, 64, 1)

    i = 0
    #save the image data into each type of defect directory 
    for batch in datagen.flow(x, batch_size = 1,
                              save_to_dir = directory, save_prefix = image_prefix, save_format = 'png'):
        i += 1
        if i > N:
            break  # otherwise the generator would loop indefinitely
    
    return

"""---------------------------------Function for adding noise to the images-------------------------------"""

def AddNoiseImage(img, folder_name, save_directory, image):
    
    """This function uses random_noise from skimage.util 
    to add gaussian, poisson, speckle and salt and pepper noise to the images (separately).
    
    Arguments:
    
    img : the augemented image data you read in
    folder_name : for example, if it is 'Mo', it means the raw data is from folder 'Mo', and the image with noise will be saved under 'Mo/Mo_Augmented'             
    save_directory : Directory where the images will be saved after adding noise            
    image : the filename of augmented image, after adding noise to this image, a new image named "noise type"+image will be saved
    
    Returns:
    
    Separate images with Gaussian, Poisson, Speckle and Salt&Pepper noise added.
    
    Raises : 
    
    Error when input data type is incorrect
    
    """
    assert type(save_directory) == str, ('Wrong Data type', 'save_directory must be a string')
    assert type(folder_name) == str, ('Wrong Data type', 'folder_name must be a string')
    assert type(image) == str, ('Wrong Data type', 'image must be a string')
    

    img_grey = rgb2gray(img)
    img_gauss = random_noise(img_grey, mode='gaussian', mean=0., var=0.01)
    img_sp = random_noise(img_grey, mode='s&p', salt_vs_pepper=0.5)
    img_poisson = random_noise(img_grey, mode='poisson')
    img_speckle = random_noise(img_grey, mode = 'speckle')
    
    imsave(save_directory+'/'+'gauss_'+image, img_gauss)
    imsave(save_directory+'/'+'s&p_'+image, img_sp)
    imsave(save_directory+'/'+'poisson'+image, img_poisson)
    imsave(save_directory+'/'+'speckle'+image, img_speckle)
    
    return 


















