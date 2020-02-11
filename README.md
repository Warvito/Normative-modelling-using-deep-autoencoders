# Normative modelling using deep autoencoders

Official script for the paper "Normative modelling using deep autoencoders: a multi-cohort study on mild cognitive impairment and Alzheimer’s disease".

[![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](https://github.com/Warvito/Normative-modelling-using-deep-autoencoders/blob/master/LICENSE)

## Abstract
Normative modelling is an emerging method for quantifying how individuals deviate from the healthy populational pattern. Several machine learning models have been implemented to develop normative models to investigate brain disorders, including regression, support vector machines and Gaussian process models. With the advance of deep learning technology, the use of deep neural networks has also been proposed. In this study, we assessed normative models based on deep autoencoders using structural neuroimaging data from patients with Alzheimer's disease (n=206) and mild cognitive impairment (n=354). We first trained the autoencoder on an independent dataset (UK Biobank dataset) with 11,034 healthy controls. Then, we estimated how each patient deviated from this norm and established which brain regions were associated to this deviation. Finally, we compared the performance of our normative model against traditional classifiers. As expected, we found that patients exhibited deviations according to the severity of their clinical condition. The model identified medial temporal regions, including the hippocampus, and the ventricular system as critical regions for the calculation of the deviation score. Overall, the normative model had comparable cross-cohort generalizability to traditional classifiers. In order to promote open science, we are making all scripts and the trained models available to the wider research community.


## Test our models online
Test our  models in this Google's colab script <a href="https://colab.research.google.com/github/Warvito/Normative-modelling-using-deep-autoencoders/blob/master/notebooks/predict.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


## Installing the dependencies and running scripts
Check out our [wiki](https://github.com/Warvito/Normative-modelling-using-deep-autoencoders/wiki) to cnfigurate your system enviroment to run our scripts on [virtualenv](https://github.com/Warvito/Normative-modelling-using-deep-autoencoders/wiki/Running-code-using-virtual-enviroment) or [Docker](https://github.com/Warvito/Normative-modelling-using-deep-autoencoders/wiki/Running-code-using-Docker).

## Script execution sequence
The sequence of our scripts and its arguments can be found in the [commands_list.sh](commands_list.sh) and [docker_command_list.sh](docker_command_list.sh) files. Check also our [wiki](https://github.com/Warvito/Normative-modelling-using-deep-autoencoders/wiki/Script-execution-sequence) with a small explanation of each file.

## Getting data
Check our [wiki](https://github.com/Warvito/Normative-modelling-using-deep-autoencoders/wiki/Getting-data) with the instructions to get the used data in this study from their original site.

## Citation
If you find this code useful for your research, please cite:

    @article {Pinaya2020.02.10.931824,
	    author = {Pinaya, Walter H. L. and Scarpazza, Cristina and Garcia-Dias, Rafael and Vieira, Sandra and Baecker, Lea and da Costa, Pedro F. and Redolfi, Alberto and Frisoni, Giovanni B. and Pievani, Michela and Calhoun, Vince D. and Sato, Jo{\~a}o R. and Mechelli, Andrea and , and ,},
	    title = {Normative modelling using deep autoencoders: a multi-cohort study on mild cognitive impairment and Alzheimer{\textquoteright}s disease},
	    elocation-id = {2020.02.10.931824},
	    year = {2020},
	    doi = {10.1101/2020.02.10.931824},
	    publisher = {Cold Spring Harbor Laboratory},
	    URL = {https://www.biorxiv.org/content/early/2020/02/11/2020.02.10.931824},
	    eprint = {https://www.biorxiv.org/content/early/2020/02/11/2020.02.10.931824.full.pdf},
	    journal = {bioRxiv}
}
