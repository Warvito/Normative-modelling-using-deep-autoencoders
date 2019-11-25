#!/usr/bin/env python3
"""
Script to create Figure 2 of the paper.
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import load_dataset

PROJECT_ROOT = Path.cwd()


def main():
    """Create elements for figure 2 of the paper"""
    # ----------------------------------------------------------------------------
    n_bootstrap = 1000
    model_name = 'supervised_aae'

    # ----------------------------------------------------------------------------
    dataset_name = 'ADNI'
    participants_path = PROJECT_ROOT / 'data' / dataset_name / 'participants.tsv'
    freesurfer_path = PROJECT_ROOT / 'data' / dataset_name / 'freesurferData.csv'
    ids_path = PROJECT_ROOT / 'outputs' / (dataset_name + '_homogeneous_ids.csv')
    adni_df = load_dataset(participants_path, ids_path, freesurfer_path)

    outputs_dir = PROJECT_ROOT / 'outputs'
    bootstrap_dir = outputs_dir / 'bootstrap_analysis'
    model_dir = bootstrap_dir / model_name

    mean_adni_list = []

    for i_bootstrap in tqdm(range(n_bootstrap)):
        bootstrap_model_dir = model_dir / '{:03d}'.format(i_bootstrap)
        output_dataset_dir = bootstrap_model_dir / dataset_name
        output_dataset_dir.mkdir(exist_ok=True)

        reconstruction_error_df = pd.read_csv(output_dataset_dir / 'reconstruction_error.csv')

        error_hc = reconstruction_error_df.loc[adni_df['Diagn'] == 1]['Reconstruction error']
        error_emci = reconstruction_error_df.loc[adni_df['Diagn'] == 27]['Reconstruction error']
        error_lmci = reconstruction_error_df.loc[adni_df['Diagn'] == 28]['Reconstruction error']
        error_ad = reconstruction_error_df.loc[adni_df['Diagn'] == 17]['Reconstruction error']

        mean_adni_list.append([error_hc.mean(), error_emci.mean(), error_lmci.mean(), error_ad.mean()])

    mean_adni_list = np.array(mean_adni_list)
    plt.hlines(range(4),
               np.percentile(mean_adni_list, 2.5, axis=0),
               np.percentile(mean_adni_list, 97.5, axis=0))

    plt.plot(np.mean(mean_adni_list, axis=0), range(4), 's', color='k')
    plt.savefig(bootstrap_dir / 'ADNI.eps', format='eps')
    plt.close()
    plt.clf()

    # ----------------------------------------------------------------------------
    dataset_name = 'FBF_Brescia'
    participants_path = PROJECT_ROOT / 'data' / dataset_name / 'participants.tsv'
    freesurfer_path = PROJECT_ROOT / 'data' / dataset_name / 'freesurferData.csv'
    ids_path = PROJECT_ROOT / 'outputs' / (dataset_name + '_homogeneous_ids.csv')
    brescia_df = load_dataset(participants_path, ids_path, freesurfer_path)

    mean_brescia_list = []

    for i_bootstrap in tqdm(range(n_bootstrap)):
        bootstrap_model_dir = model_dir / '{:03d}'.format(i_bootstrap)
        output_dataset_dir = bootstrap_model_dir / dataset_name
        output_dataset_dir.mkdir(exist_ok=True)

        reconstruction_error_df = pd.read_csv(output_dataset_dir / 'reconstruction_error.csv')

        error_hc = reconstruction_error_df.loc[brescia_df['Diagn'] == 1]['Reconstruction error']
        error_mci = reconstruction_error_df.loc[brescia_df['Diagn'] == 18]['Reconstruction error']
        error_ad = reconstruction_error_df.loc[brescia_df['Diagn'] == 17]['Reconstruction error']

        mean_brescia_list.append([error_hc.mean(), error_mci.mean(), error_ad.mean()])

    mean_brescia_list = np.array(mean_brescia_list)
    plt.hlines(range(3),
               np.percentile(mean_brescia_list, 2.5, axis=0),
               np.percentile(mean_brescia_list, 97.5, axis=0))

    plt.plot(np.mean(mean_brescia_list, axis=0), range(3), 's', color='k')
    plt.savefig(bootstrap_dir / 'TOMC.eps', format='eps')
    plt.close()
    plt.clf()

    # ----------------------------------------------------------------------------
    dataset_name = 'OASIS1'
    participants_path = PROJECT_ROOT / 'data' / dataset_name / 'participants.tsv'
    freesurfer_path = PROJECT_ROOT / 'data' / dataset_name / 'freesurferData.csv'
    ids_path = PROJECT_ROOT / 'outputs' / (dataset_name + '_homogeneous_ids.csv')
    oasis1_df = load_dataset(participants_path, ids_path, freesurfer_path)

    mean_oasis1_list = []

    for i_bootstrap in tqdm(range(n_bootstrap)):
        bootstrap_model_dir = model_dir / '{:03d}'.format(i_bootstrap)
        output_dataset_dir = bootstrap_model_dir / dataset_name
        output_dataset_dir.mkdir(exist_ok=True)

        reconstruction_error_df = pd.read_csv(output_dataset_dir / 'reconstruction_error.csv')

        error_hc = reconstruction_error_df.loc[oasis1_df['Diagn'] == 1]['Reconstruction error']
        error_ad = reconstruction_error_df.loc[oasis1_df['Diagn'] == 17]['Reconstruction error']

        mean_oasis1_list.append([error_hc.mean(), error_ad.mean()])

    mean_oasis1_list = np.array(mean_oasis1_list)
    plt.hlines(range(2),
               np.percentile(mean_oasis1_list, 2.5, axis=0),
               np.percentile(mean_oasis1_list, 97.5, axis=0))

    plt.plot(np.mean(mean_oasis1_list, axis=0), range(2), 's', color='k')
    plt.savefig(bootstrap_dir / 'OASIS1.eps', format='eps')
    plt.close()
    plt.clf()


if __name__ == "__main__":
    main()
