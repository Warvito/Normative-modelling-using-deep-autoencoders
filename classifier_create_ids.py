#!/usr/bin/env python3
"""Script to create the ids of the used to train and to evaluate the classifiers across the bootstrap method."""
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils import load_demographic_data

PROJECT_ROOT = Path.cwd()


def main(dataset_name, disease_label):
    """Create the ids for the classifier analysis"""
    # ----------------------------------------------------------------------------------------
    n_bootstrap = 1000

    participants_path = PROJECT_ROOT / 'data' / dataset_name / 'participants.tsv'

    hc_label = 1

    # ----------------------------------------------------------------------------
    # Create experiment's output directory
    ids_path = PROJECT_ROOT / 'outputs' / (dataset_name + '_homogeneous_ids.csv')

    classifier_dir = PROJECT_ROOT / 'outputs' / 'classifier_analysis'
    classifier_dir.mkdir(exist_ok=True)
    classifier_dataset_dir = classifier_dir / dataset_name
    classifier_dataset_dir.mkdir(exist_ok=True)

    classifier_dataset_analysis_dir = classifier_dataset_dir / '{:02d}_vs_{:02d}'.format(hc_label, disease_label)
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

    for i_bootstrap in tqdm(range(n_bootstrap)):
        bootstrap_ids = dataset_df.sample(n=n_sub, replace=True, random_state=i_bootstrap)

        ids_filename = 'homogeneous_bootstrap_{:03d}_train.csv'.format(i_bootstrap)
        bootstrap_ids.to_csv(ids_dir / ids_filename, index=False)

        not_sampled = ~dataset_df['Image_ID'].isin(bootstrap_ids['Image_ID'])
        bootstrap_ids_test = dataset_df[not_sampled]
        ids_filename = 'homogeneous_bootstrap_{:03d}_test.csv'.format(i_bootstrap)
        bootstrap_ids_test.to_csv(ids_dir / ids_filename, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--dataset_name',
                        dest='dataset_name',
                        help='Dataset name to create the ids for bootstrap method.')
    parser.add_argument('-L', '--disease_label',
                        dest='disease_label',
                        help='Disease label to create the ids for bootstrap method.',
                        type=int)
    args = parser.parse_args()

    main(args.dataset_name, args.disease_label)
