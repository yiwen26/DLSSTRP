from __future__ import absolute_import, division, print_function
import numpy as np
import keras
import pathmagic # noqa
from defectfinder import classactmap as cmap

"""Testing functions for class activation modules"""


def test_image_preprocessing_cam():
    """
    Testing the input image and its size for correct data type
    """
    try:
        image_data = ()
        image_size = int
        cmap.image_preprocessing_cam(image_data, image_size)

    except (Exception):
        pass

    else:
        raise Exception('Error not handled')

    return


def test_get_predictions():
    """
    Testing the input image and its size for correct data type
    """
    try:
        image = ()
        model = keras.models
        cmap.get_predictions(model, image)

    except (Exception):
        pass

    else:
        raise Exception('Error not handled')

    return


def test_get_SoftmaxWeights():
    """
    Testing if the model input is correct
    """
    try:
        model = keras.models
        cmap.get_SoftmaxWeights(model)

    except (Exception):
        pass

    else:
        raise Exception('Error not handled')

    return


def test_get_defects():
    """
    Testing for correct input data types
    """
    try:
        defects = ()
        n = np.float
        cmap.get_defects(defects, n)

    except (Exception):
        pass

    else:
        raise Exception('Error not handled')

    return
