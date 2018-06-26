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

`defectfinder` is a software package that allows the user to develop a convolutional neural network (CNN) based framework to automate the localization, classification and visualization of defects in 2D materials from dynamic STEM data. In order to gain physically relevant classification and insight into the defects in these materials, a CNN needs to be trained on the theoretically simulated defects in arising in the materials. To account for effects such as experimental noise and generalizability of the model, the theoretical images need to augmented along with different kinds of random noise. `preprocessing` module in the software allows the user to do the same. After the preprocessing, a convolutional-neural-network can be trained on these theoretical augmented and noisy images. The parameters involved in network have a complex interplay between them and therefore, the ideal optimized parameters need to be used for the training. `gridsearch` module allows the user to optimize the parameters for the training. The results from the CNN training can be visualized with the help of class activation maps using the `classactmap` module. Class activation maps provide the user with the insight into where the network is focusing on while making predictions and whether it is physically relevant or not. Additionally, this also allows the user to accurately locate the defect coordinates.

In order to analyze experimental datasets, `extract_predict` module can used. This particular module allows the user to localize and extract the defects frame by frame using FFT subtraction and lattice periodicity and predict the defect type for all the localized defects. The predicted defects can visualized with type and specific coordinates using class activation maps from `classactmap` module. Furthermore, the interplay between the different defect types and as a function of time can also be analyzed using this module.

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
### Optional
* pydm3reader (for reading in experimental data - download link: https://microscopies.med.univ-tours.fr/?page_id=260)

## Repository Outlines
### Data
* Multislice STEM data (with augmentation and noise added) for all individual defects in WS~2~ can be found in the **Multislice** folder
* Models trained on WS~2~ theoretical defects data can be found in the **Trained_Models** folder
* **defect_list** folder contains the localized defect data for all the frames in the dynamic STEM movie
* **Figures** folder contains all the figures generated from the code in the **Jupyter_notebooks**
### Example
#### **Jupyter_notebooks** folder contains example notebooks implementing different modules from the software
* *GenerateTrainingData* notebook demonstrates the application of `preprocessing` module to generate images with augmentation and noise.
* *SimpleCNN* notebook shows how to use a simple convolutional neural network to train on the generated images and also demonstrates the use of `classactmap` module to visualize the class activation maps using the previously trained model.
* *Hyperparameters Tuning* notebook demonstrates the use of `gridsearch` to optimize the training parameters for the convolutional neural network.
* *Extract_Defect* notebook uses `extract_predict` module and demonstrates to localize, extract and predict defects based on models trained on theoretical Multislice data. It also uses class activation maps for visualization of defect types and coordinates. Furthermore, it demonstrates different analysis of defects such as time evolution (or framewise) of defects.
* *VideoFrame* notebook shows how to reconstruct the dynamic STEM movie with labeled defect predictions.
### Use Case
![alt text](https://github.com/yiwen26/DLSSTRP/blob/master/UseCase/use%20case.png)

## License
This is currently a research project, and we do not plan to commercialize this, this project is under the permissive MIT license. If anything changes, we will be sure to update accordingly. If you do happen to want to use any parts of this project, please do give reference. For more details, please read LICENSE.md
