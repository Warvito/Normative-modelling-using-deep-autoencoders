# Normative modelling using deep autoencoders

Official script for the paper "Normative modelling using deep autoencoders: a multi-sample study on mild cognitive impairment and Alzheimer’s disease".

[![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](https://github.com/Warvito/Normative-modelling-using-deep-autoencoders/blob/master/LICENSE)



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

#### 1. Preprocessing
1. clean_biobank1_data.py
2. combine_sites_data.py
3. clean_clinical_data.py
4. demographic_balancing_adni_data.py
5. demographic_balancing_tomc_data.py

#### 2a. Bootstrap analysis
Analysis where we measure the performance of the normative approach using a bootstrap method. 
These scripts obtain the results from sections 3.1, 3.2, 3.3 of the paper. 
1. bootstrap_create_ids.py
2. bootstrap_train_aae_supervised.py
3. bootstrap_test_aae_supervised.py
4. bootstrap_group_analysis_2x2.py
5. bootstrap_create_figures.py

#### 2b. Classifier analysis
Analysis where we measure the performance of the binary classification (using RVM).
To estimate the performance of the model, we used the .632+ bootstrap method.
These scripts obtain the results from sections 3.4 of the paper. 
1. classifier_bootstrap_create_ids.py
2. classifier_bootstrap_train.py
3. classifier_bootstrap_analysis.py

#### 2c. Miscellaneous 
1. univariate_analysis.py

## Citation
If you find this code useful for your research, please cite:

    @article{}