# Normative modelling using deep autoencoders

Official script for the paper "Normative modelling using deep autoencoders: a multi-cohort study on mild cognitive impairment and Alzheimer’s disease".

[![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](https://github.com/Warvito/Normative-modelling-using-deep-autoencoders/blob/master/LICENSE)

## Abstract
Normative modelling is an emerging method that allows quantifying how individuals deviate from the population pattern and recently have been used to analyse neuroimaging data in brain disorders. Many machine learning models have been adapted for this task, including regressions, support vector machines and Gaussian process. With the recent success of deep learning, the use of deep neural networks has also been proposed. In this study, we assessed normative models based on deep autoencoders using data from patients with Alzheimer’s disease (n=79) and mild cognitive impairment (n=270). In our analysis, we train the autoencoder on an independent dataset (UK Biobank dataset) with 11,032 healthy controls. Then, we verified how each patient deviates from the norm, which brain regions were responsible for this deviation, and how this model performs in a classification task. We verified that the subjects presented deviations according to the severity of their clinical condition. The model pointed regions from the medial temporal lobe, including the hippocampus and entorhinal cortex, as important for the calculation of the deviation score. Finally, the normative model had similar performance compared to traditional classifiers.


## Test our models online
Test our  models in this Google's colab script <a href="https://colab.research.google.com/github/Warvito/Normative-modelling-using-deep-autoencoders/blob/master/notebooks/predict_deviation_bootstrap.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


## Requirements
- Python 3
- [Tensorflow 2.0](https://www.tensorflow.org/)
- [Numpy](http://www.numpy.org/)
- [Matplotlib](https://matplotlib.org/)


## Installing the dependencies
Install virtualenv and creating a new virtual environment:

    pip install virtualenv
    virtualenv -p /usr/bin/python3 ./venv

Install dependencies

    pip install -r requirements.txt


## Script execution sequence
The sequence of our codes and its arguments can be found in the commands_list.sh file. 
The following list indicates the different phases and its scripts.

#### 0. Getting data
Script for internal use (Machine Learning in Mental Health Lab only).
1. [download_datasets.py](https://github.com/Warvito/Normative-modelling-using-deep-autoencoders/blob/master/download_datasets.py)
2. [combine_sites_data.py](https://github.com/Warvito/Normative-modelling-using-deep-autoencoders/blob/master/combine_sites_data.py)

#### 1. Preprocessing
1. [clean_biobank1_data.py](https://github.com/Warvito/Normative-modelling-using-deep-autoencoders/blob/master/clean_biobank1_data.py)
3. [clean_clinical_data.py](https://github.com/Warvito/Normative-modelling-using-deep-autoencoders/blob/master/clean_clinical_data.py)
4. [demographic_balancing_adni_data.py](https://github.com/Warvito/Normative-modelling-using-deep-autoencoders/blob/master/demographic_balancing_adni_data.py)
5. [demographic_balancing_tomc_data.py](https://github.com/Warvito/Normative-modelling-using-deep-autoencoders/blob/master/demographic_balancing_tomc_data.py)
6. [demographic_balancing_oasis1_data.py](https://github.com/Warvito/Normative-modelling-using-deep-autoencoders/blob/master/demographic_balancing_oasis1_data.py)

#### 2a. Bootstrap analysis
Analysis where we measure the performance of the normative approach using a bootstrap method. 
These scripts obtain the results from sections 3.1, 3.2, 3.3 of the paper. 
1. [bootstrap_create_ids.py](https://github.com/Warvito/Normative-modelling-using-deep-autoencoders/blob/master/bootstrap_create_ids.py)
2. [bootstrap_train_aae_supervised.py](https://github.com/Warvito/Normative-modelling-using-deep-autoencoders/blob/master/bootstrap_train_aae_supervised.py)
3. [bootstrap_test_aae_supervised.py](https://github.com/Warvito/Normative-modelling-using-deep-autoencoders/blob/master/bootstrap_test_aae_supervised.py)
4. [bootstrap_group_analysis_1x1.py](https://github.com/Warvito/Normative-modelling-using-deep-autoencoders/blob/master/bootstrap_group_analysis_1x1.py)
5. [bootstrap_create_figures.py](https://github.com/Warvito/Normative-modelling-using-deep-autoencoders/blob/master/bootstrap_create_figures.py)
6. [bootstrap_hypothesis_test.py](https://github.com/Warvito/Normative-modelling-using-deep-autoencoders/blob/master/bootstrap_hypothesis_test.py)

#### 2b. Classifier analysis
Analysis where we measure the performance of the binary classification (using RVM).
To estimate the performance of the model, we used the .632+ bootstrap method.
These scripts obtain the results from sections 3.4 of the paper. 
1. [classifier_create_ids.py](https://github.com/Warvito/Normative-modelling-using-deep-autoencoders/blob/master/classifier_create_ids.py)
2. [classifier_train.py](https://github.com/Warvito/Normative-modelling-using-deep-autoencoders/blob/master/classifier_train.py)
3. [classifier_group_analysis_1x1.py](https://github.com/Warvito/Normative-modelling-using-deep-autoencoders/blob/master/classifier_group_analysis_1x1.py)
4. [bootstrap_normative_vs_classifier.py](https://github.com/Warvito/Normative-modelling-using-deep-autoencoders/blob/master/bootstrap_normative_vs_classifier.py)

#### 2c. Miscellaneous 
1. [univariate_analysis.py](https://github.com/Warvito/Normative-modelling-using-deep-autoencoders/blob/master/univariate_analysis.py)

## Citation
If you find this code useful for your research, please cite:

    @article{}
