# Defects Discoverers (defectfinder)
## Deep Learning of Solid-State Transformations and Reaction Pathways in 2D Materials
[![Build Status](https://travis-ci.org/SarthakJariwala/DLSSTRP.svg?branch=master)](https://travis-ci.org/SarthakJariwala/DLSSTRP) [![PyPI version](https://badge.fury.io/py/defectfinder.svg)](https://badge.fury.io/py/defectfinder)

Project with `Oak Ridge National Lab`
* Group members: Sarthak, Jimin and Yiwen
* Mentor: Dr. Maxim Ziatdinov

## Download Software
This package can be installed by running `pip install defectfinder` in the command line

## Software Description
Recent advances in scanning transmission electron microscopy (STEM) have allowed unprecedented insight into the elementary mechanisms behind the solid-state phase transformations and reactions. However, the ability to quickly acquire large, high-resolution datasets has created a challenge for rapid physics-based analysis of STEM images and movies. 

`defectfinder` is a software package that allows the user to develop a convolutional-neural-network (cNN) based framework to automate the localization, classification and visualization of defects in 2D materials from dynamic STEM data. In order to gain physically relevant classification and insight into the defects in these materials, a cNN needs to be trained on the theoretically simulated defects in arising in the materials. To account for effects such as experimental noise and generalizability of the model, the theoretical images need to augmented along with different kinds of random noise. `preprocessing` module in the software allows the user to do the same. After the preprocessing, a convolutional-neural-network can be trained on these theoretical augmented and noisy images. The parameters involved in network have a complex interplay between them and therefore, the ideal optimized parameters need to be used for the training. `gridsearch` module allows the user to optimize the parameters for the training. The results from the cNN training can be visualized with the help of class activation maps using the `classactmap` module. Class activation maps allow the user with the insight into where the network is focusing on while making predictions and whether it is physically relevant or not. Additionally, this also allows the user to accurately locate the defect coordinates.

(experimental data analysis details to be added) - extract and predict module...

## Software Dependencies
### Language
* Python 3 or greater
### Packages
* keras
* numpy
* scipy
* scikit-learn
* scikit-image
* opencv-Python

## License
This is currently a research project, and we do not plan to commercialize this, this project is under the permissive MIT license. If anything changes, we will be sure to update accordingly. If you do happen to want to use any parts of this project, please do give reference. For more details, please read LICENSE.md

## Repository Outlines
### Data
simulated STEM data on individual defects of WS_2
### Docs
Documentations about this project. (coming soon)
### Use Case
![alt text](https://github.com/yiwen26/DLSSTRP/blob/master/UseCase/use%20case.png)

### Poster
