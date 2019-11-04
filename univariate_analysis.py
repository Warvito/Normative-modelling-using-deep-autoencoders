"""
Script to perform the mass-univariate analysis

"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from utils import COLUMNS_NAME, load_dataset

PROJECT_ROOT = Path.cwd()

def cliff_delta(X, Y):
    lx = len(X)
    ly = len(Y)
    mat = np.zeros((lx, ly))
    for i in range(0, lx):
        for j in range(0, ly):
            if X[i] > Y[j]:
                mat[i, j] = 1
            elif Y[j] > X[i]:
                mat[i, j] = -1

    return (np.sum(mat)) / (lx * ly)

def main():
    # ----------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'
    dataset_name = 'FBF_Brescia'

    participants_path = PROJECT_ROOT / 'data' / 'datasets' / dataset_name / 'participants.tsv'
    freesurfer_path = PROJECT_ROOT / 'data' / 'datasets' / dataset_name / 'freesurferData.csv'

    hc_label = 1
    disease_label = 18
    # ----------------------------------------------------------------------------
    # Create directories structure
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    ids_path = experiment_dir / (dataset_name + '_homogeneous_ids.csv')

    univariate_dir = experiment_dir / 'univariate_analysis'
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
    main()
