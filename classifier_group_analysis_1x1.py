#!/usr/bin/env python3
"""Script to get the classification performance.

References:
    https://stats.stackexchange.com/questions/96739/what-is-the-632-rule-in-bootstrapping
    https://github.com/rasbt/mlxtend/blob/9c044a920c31054fa106fb028e9115a3bd852cf8/mlxtend/evaluate/bootstrap_point632.py
"""
import argparse
from pathlib import Path
import random as rn

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from utils import load_dataset

PROJECT_ROOT = Path.cwd()


def no_information_rate(targets, predictions, loss_fn):
    """Calculate the proportion of overfitting."""
    combinations = np.array(np.meshgrid(targets, predictions)).reshape(-1, 2)
    return loss_fn(combinations[:, 0], combinations[:, 1])


def main(dataset_name, disease_label):
    """Calculate the performance of the AUC-ROC for the classifier."""
    # ----------------------------------------------------------------------------
    n_bootstrap = 1000

    participants_path = PROJECT_ROOT / 'data' / dataset_name / 'participants.tsv'
    freesurfer_path = PROJECT_ROOT / 'data' / dataset_name / 'freesurferData.csv'

    hc_label = 1

    # ----------------------------------------------------------------------------
    # Set random seed
    random_seed = 42
    np.random.seed(random_seed)
    rn.seed(random_seed)

    classifier_dir = PROJECT_ROOT / 'outputs' / 'classifier_analysis'
    classifier_dataset_dir = classifier_dir / dataset_name
    classifier_dataset_analysis_dir = classifier_dataset_dir / '{:02d}_vs_{:02d}'.format(hc_label, disease_label)

    ids_path = PROJECT_ROOT / 'outputs' / (dataset_name + '_homogeneous_ids.csv')

    auc_bootstrap_train = np.load(classifier_dataset_analysis_dir / 'aucs_train.npy')
    auc_bootstrap_test = np.load(classifier_dataset_analysis_dir / 'aucs_test.npy')

    # ----------------------------------------------------------------------------
    bootstrap = []
    for i_bootstrap in tqdm(range(n_bootstrap)):
        predictions = pd.read_csv(
            classifier_dataset_analysis_dir / 'predictions' / 'homogeneous_bootstrap_{:03d}_prediction.csv'.format(
                i_bootstrap))
        dataset_df = load_dataset(participants_path, ids_path, freesurfer_path)
        dataset = pd.merge(predictions, dataset_df, on='Image_ID')
        dataset['Diagn'] = dataset['Diagn'].map({hc_label: 0, disease_label: 1})

        # ----------------------------------------------------------------------------
        # Measuring the performance using .632+ bootstrap method
        auc_resubstitution = auc_bootstrap_train[i_bootstrap]
        auc_out_of_bag = auc_bootstrap_test[i_bootstrap]
        gamma = no_information_rate(dataset['Diagn'].values,
                                    dataset['predictions'].values,
                                    roc_auc_score)
        R = (- (auc_out_of_bag - auc_resubstitution)) / (gamma - (1 - auc_out_of_bag))
        w = 0.632 / (1 - 0.368 * R)

        bootstrap.append((w * auc_out_of_bag + (1 - w) * auc_resubstitution))

    all_aucs = pd.DataFrame(columns=['AUCS'], data=bootstrap)
    all_aucs.to_csv(classifier_dataset_analysis_dir / 'all_AUCs.csv', index=False)

    results = pd.DataFrame(columns=['Measure', 'Value'])
    results = results.append({'Measure': 'mean', 'Value': np.mean(bootstrap)}, ignore_index = True)
    results = results.append({'Measure': 'upper_limit', 'Value': np.percentile(bootstrap, 97.5)}, ignore_index = True)
    results = results.append({'Measure': 'lower_limit', 'Value': np.percentile(bootstrap, 2.5)}, ignore_index = True)
    results.to_csv(classifier_dataset_analysis_dir / 'AUCs_summary.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--dataset_name',
                        dest='dataset_name',
                        help='Dataset name to perform the group analysis.')
    parser.add_argument('-L', '--disease_label',
                        dest='disease_label',
                        help='Disease label to perform the group analysis.',
                        type=int)
    args = parser.parse_args()

    main(args.dataset_name, args.disease_label)