#!/usr/bin/env python3
"""Script to perform the mass-univariate analysis"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from utils import COLUMNS_NAME, load_dataset, cliff_delta

PROJECT_ROOT = Path.cwd()


def main(dataset_name, disease_label):
    # ----------------------------------------------------------------------------
    participants_path = PROJECT_ROOT / 'data' / dataset_name / 'participants.tsv'
    freesurfer_path = PROJECT_ROOT / 'data' / dataset_name / 'freesurferData.csv'

    hc_label = 1
    # ----------------------------------------------------------------------------
    # Create directories structure
    # ----------------------------------------------------------------------------
    ids_path = PROJECT_ROOT / 'outputs' / (dataset_name + '_homogeneous_ids.csv')
    univariate_dir = PROJECT_ROOT / 'outputs' / 'univariate_analysis'
    univariate_dir.mkdir(exist_ok=True)

    # ----------------------------------------------------------------------------
    # Loading data
    clinical_df = load_dataset(participants_path, ids_path, freesurfer_path)

    x_dataset = clinical_df[COLUMNS_NAME].values

    tiv = clinical_df['EstimatedTotalIntraCranialVol'].values
    tiv = tiv[:, np.newaxis]

    clinical_df[COLUMNS_NAME] = (np.true_divide(x_dataset, tiv)).astype('float32')

    results = pd.DataFrame()

    for region in COLUMNS_NAME:

        statistic, pvalue = stats.mannwhitneyu(clinical_df[clinical_df['Diagn']==hc_label][region],
                                               clinical_df[clinical_df['Diagn']==disease_label][region])

        effect_size = cliff_delta(clinical_df[clinical_df['Diagn']==hc_label][region].values,
                                  clinical_df[clinical_df['Diagn']==disease_label][region].values)

        results = results.append({'regions': region, 'effect size': effect_size, 'p-value': pvalue}, ignore_index=True)

    results.to_csv(univariate_dir / '{}_{}_vs_{}.csv'.format(dataset_name, hc_label, disease_label), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--dataset_name',
                        dest='dataset_name',
                        help='Dataset name to perform univarate analysis.')
    parser.add_argument('-L', '--disease_label',
                        dest='disease_label',
                        help='Disease label to perform univarate analysis.',
                        type=int)
    args = parser.parse_args()

    main(args.dataset_name, args.disease_label)