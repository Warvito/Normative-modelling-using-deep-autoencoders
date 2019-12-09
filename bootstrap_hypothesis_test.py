#!/usr/bin/env python3
"""Script to perform the hypothesis test over groups."""
import argparse
from pathlib import Path
from itertools import combinations

import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm

from utils import load_dataset

PROJECT_ROOT = Path.cwd()


def main(dataset_name, label_list):
    """Perform the group analysis."""
    # ----------------------------------------------------------------------------
    n_bootstrap = 1000

    model_name = 'supervised_aae'

    participants_path = PROJECT_ROOT / 'data' / dataset_name / 'participants.tsv'
    freesurfer_path = PROJECT_ROOT / 'data' / dataset_name / 'freesurferData.csv'
    ids_path = PROJECT_ROOT / 'outputs' / (dataset_name + '_homogeneous_ids.csv')
    # ----------------------------------------------------------------------------
    clinical_df = load_dataset(participants_path, ids_path, freesurfer_path)
    # ----------------------------------------------------------------------------
    bootstrap_dir = PROJECT_ROOT / 'outputs' / 'bootstrap_analysis'
    model_dir = bootstrap_dir / model_name

    reconstruction_error_list_df = pd.DataFrame(clinical_df['Participant_ID'])
    clinical_df = clinical_df.set_index('Participant_ID')

    for i_bootstrap in tqdm(range(n_bootstrap)):
        bootstrap_model_dir = model_dir / '{:03d}'.format(i_bootstrap)

        output_dataset_dir = bootstrap_model_dir / dataset_name
        output_dataset_dir.mkdir(exist_ok=True)

        # ----------------------------------------------------------------------------
        reconstruction_error_df = pd.read_csv(output_dataset_dir / 'reconstruction_error.csv')
        reconstruction_error_list_df['Reconstruction error {:d}'.format(i_bootstrap)] = reconstruction_error_df[
            'Reconstruction error'].values

    reconstruction_error_list_df = reconstruction_error_list_df.set_index('Participant_ID')

    hypothesis_df = pd.DataFrame()
    for group_labels in combinations(label_list, 2):
        mean_group1 = np.mean(reconstruction_error_list_df.loc[clinical_df['Diagn'] == group_labels[0]].values, axis=0)
        mean_group2 = np.mean(reconstruction_error_list_df.loc[clinical_df['Diagn'] == group_labels[1]].values, axis=0)

        t_stats, p_value = stats.ttest_ind(mean_group1, mean_group2)

        data = {'comparison':'{}_vs_{}'.format(group_labels[0], group_labels[1]),
                'p-value': p_value,
                't_stats': t_stats}

        hypothesis_df = hypothesis_df.append(data, ignore_index=True)

    hypothesis_df.to_csv(bootstrap_dir / dataset_name / 'hypothesis_test.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--dataset_name',
                        dest='dataset_name',
                        help='Dataset name to perform group analysis.')
    parser.add_argument('-L', '--label_list',
                        dest='label_list',
                        nargs='+',
                        help='List of labels to perform the analysis.',
                        type=int)
    args = parser.parse_args()

    main(args.dataset_name, args.label_list)
