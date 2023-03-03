# Scientifc Reports 23 Source Code
 Application of machine learning techniques for identifying soil-dwelling fungi and chromista

# Image Classification Scripts

This repository contains scripts for image classification using various algorithms such as kNN, SVM, Random Forest, MLP, ELM, and Regularized ELM. The scripts were developed using MATLAB.

## Scripts with Description

- `main.m` - main function to run `makeMasks2`, `makeFeaturesFromImagesAndMasks`, and `makeClassification`
- `makeMasks2.m` - preprocesses images and segments them, creating an image mask with image scaling functionality. Takes input images and masks paths.
- `makeFeaturesFromImagesAndMasks.m` - calculates feature values based on images and masks. Takes input images and masks paths and an output data file, which contains all feature values that are saved in the data folder. The function also returns the calculated features.
- `makeClassification.m` - takes a `dataFile.mat` file with calculated features by `makeFeaturesFromImagesAndMasks` and applies the given feature selection method.
- `EML.m` - Extreme Learning Machine function that takes as input `CVSet`, number of neurons, and type of activation function on hidden layer. Returns a confusion matrix `arr`, T2 confusion matrix with labels, and `t` that gives info about the time of learning the net.
- `GLCMFeatures.m` - calculates GLCM features separately for Red, Green, and Blue color channels of RGB color space.
- `GLRLMFeatures.m` - calculates GLRLM features for the original image in grayscale.
- `kNNAlogrithm.m` - kNN classifier that takes `CVSets` and `k` parameter. Returns confusion matrix `arr`, T2 as confusion matrix with labels, and `t` as the time of learning.
- `makeASummary.m` - a helper script for making a summary of the classifier's performance.
- `makeCVSets.m` - creates Cross Validation sets.
- `MLP.m` - multilayer Perceptron function that takes as input `CVSet` and topology of the net. Returns a confusion matrix `arr`, T2 confusion matrix with labels, and `t` that gives info about the time of learning the net.
- `randomForest.m` - Random Forest function that takes as input `CVSet` and number of trees. Returns a confusion matrix `arr`, T2 confusion matrix with labels, and `t` that gives info about the time of learning.
- `RELM.m` - Regularized Extreme Learning Machine function that takes as input `CVSet`, number of neurons, and type of activation function on hidden layer. Returns a confusion matrix `arr`, T2 confusion matrix with labels, and `t` that gives info about the time of learning the net.
- `svmMulti.m` - Support Vector Machine function that takes as input `CVSet`. Returns confusion matrix `arr`, T2 confusion matrix with labels, and `t` that gives info about the time of learning.

## Folders
- `Folder in` - input dataset
- `Folder out` - output image masks (in research original 2 has been applied)
- `Folder data` - supplementary files like calculated datasets as a result of running `makeFeaturesFromImagesAndMasks` (csv subfolder) and in mat subfolder contains features headings.
- `Folder resources` - external libraries used.
- `Results` - all results of calculations.
