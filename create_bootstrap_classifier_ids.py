"""
Script to create gender-homogeneous bootstrap datasets to feed into create_h5_bootstrap script;
Creates 50 bootstrap samples with increasing size
"""
from pathlib import Path

import pandas as pd
import numpy as np

from utils import load_demographic_data

PROJECT_ROOT = Path.cwd()


def main():
    """"""
    # ----------------------------------------------------------------------------------------
    n_bootstrap = 1000

    experiment_name = 'biobank_scanner1'
    dataset_name = 'PPMI'

    participants_path = PROJECT_ROOT / 'data' / 'datasets' / dataset_name / 'participants.tsv'

    hc_label = 1
    disease_label = 24

    # ----------------------------------------------------------------------------
    # Create experiment's output directory
    ids_path = PROJECT_ROOT / 'outputs' / experiment_name / (dataset_name + '_homogeneous_ids.csv')

    individual_dir = PROJECT_ROOT / 'outputs' / experiment_name / 'classifier_analysis'
    individual_dir.mkdir(exist_ok=True)
    individual_dataset_dir = individual_dir / dataset_name
    individual_dataset_dir.mkdir(exist_ok=True)

    classifier_dataset_analysis_dir = individual_dataset_dir / '{:02d}_vs_{:02d}'.format(hc_label, disease_label)
    classifier_dataset_analysis_dir.mkdir(exist_ok=True)

    ids_dir = classifier_dataset_analysis_dir / 'ids'
    ids_dir.mkdir(exist_ok=True)

    # ----------------------------------------------------------------------------------------
    # Set random seed for random sampling of subjects
    np.random.seed(42)


    # ----------------------------------------------------------------------------------------
    dataset_df = load_demographic_data(participants_path, ids_path)

    dataset_df = dataset_df.loc[(dataset_df['Diagn'] == hc_label) | (dataset_df['Diagn'] == disease_label)]
    dataset_df = dataset_df.reset_index(drop=True)
    dataset_df = dataset_df[['Image_ID']]

    n_sub = len(dataset_df)

    for i_bootstrap in range(n_bootstrap):
        bootstrap_ids = dataset_df.sample(n=n_sub, replace=True)
        ids_filename = 'homogeneous_bootstrap_{:03d}.csv'.format(i_bootstrap)
        bootstrap_ids.to_csv(ids_dir / ids_filename, index=False)


if __name__ == "__main__":
    main()
