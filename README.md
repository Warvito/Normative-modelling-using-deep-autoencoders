# Normative modelling using deep autoencoders

Official script for the paper "Normative modelling using deep autoencoders: a multi-cohort study on mild cognitive impairment and Alzheimer’s disease".

[![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](https://github.com/Warvito/Normative-modelling-using-deep-autoencoders/blob/master/LICENSE)

## Abstract
Normative modelling is an emerging method that allows quantifying how individuals deviate from the population pattern and recently have been used to analyse neuroimaging data in brain disorders. Many machine learning models have been adapted for this task, including regressions, support vector machines and Gaussian process. With the recent success of deep learning, the use of deep neural networks has also been proposed. In this study, we assessed normative models based on deep autoencoders using data from patients with Alzheimer’s disease (n=79) and mild cognitive impairment (n=270). In our analysis, we train the autoencoder on an independent dataset (UK Biobank dataset) with 11,032 healthy controls. Then, we verified how each patient deviates from the norm, which brain regions were responsible for this deviation, and how this model performs in a classification task. We verified that the subjects presented deviations according to the severity of their clinical condition. The model pointed regions from the medial temporal lobe, including the hippocampus and entorhinal cortex, as important for the calculation of the deviation score. Finally, the normative model had similar performance compared to traditional classifiers.


## Test our models online
Test our  models in this Google's colab script <a href="https://colab.research.google.com/github/Warvito/Normative-modelling-using-deep-autoencoders/blob/master/notebooks/predict_deviation_bootstrap.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


## Installing the dependencies and running scripts
Check out our [wiki](https://github.com/Warvito/Normative-modelling-using-deep-autoencoders/wiki) to cnfigurate your system enviroment to run our scripts on [virtualenv](https://github.com/Warvito/Normative-modelling-using-deep-autoencoders/wiki/Running-code-using-virtual-enviroment) or [Docker](https://github.com/Warvito/Normative-modelling-using-deep-autoencoders/wiki/Running-code-using-Docker).

## Script execution sequence
The sequence of our scripts and its arguments can be found in the [commands_list.sh](commands_list.sh) and [docker_command_list.sh](docker_command_list.sh) files. Check also our [wiki](https://github.com/Warvito/Normative-modelling-using-deep-autoencoders/wiki/Script-execution-sequence) with a small explanation of each file.

## Getting data
Check our [wiki](https://github.com/Warvito/Normative-modelling-using-deep-autoencoders/wiki/Getting-data) with the instructions to get the used data in this study from their original site.

## Citation
If you find this code useful for your research, please cite:

    @article{}
