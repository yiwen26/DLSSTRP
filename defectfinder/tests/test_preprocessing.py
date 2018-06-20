from __future__ import absolute_import, division, print_function
import pathmagic  # noqa
from defectfinder import preprocessing as pre

"""Testing functions for preprocessing module"""


def test_AugmentImage():
    """
    Testing the if the input data type for AugmentImage function is correct
    """
    try:
        path = ()
        folder_name = ()
        image_prefix = int
        N = str
        pre.AugmentImage(img, path, folder_name, image_prefix, N)

    except (Exception):
        pass

    else:
        raise Exception('Error not handled')

    return


def test_AddNoiseImage():
    """
    Testing the if the input data type for AddNoiseImage function is correct
    """
    try:
        folder_name = ()
        save_directory = int
        image = {}
        pre.AddNoiseImage(img, folder_name, save_directory, image)

    except (Exception):
        pass

    else:
        raise Exception('Error not handled')

    return
