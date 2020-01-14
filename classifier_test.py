#!/usr/bin/env python3
"""Script to get the classification performance."""
import argparse
from pathlib import Path
import random as rn

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from joblib import load

from utils import COLUMNS_NAME, load_dataset

PROJECT_ROOT = Path.cwd()


def main(dataset_name, disease_label, evaluated_dataset):
    """Calculate the performance of the classifier in each iteration of the bootstrap method."""
    # ----------------------------------------------------------------------------
    n_bootstrap = 1000

    participants_path = PROJECT_ROOT / 'data' / evaluated_dataset / 'participants.tsv'
    freesurfer_path = PROJECT_ROOT / 'data' / evaluated_dataset / 'freesurferData.csv'

    outputs_dir = PROJECT_ROOT / 'outputs'
    ids_path = outputs_dir / (evaluated_dataset + '_homogeneous_ids.csv')

    hc_label = 1

    # ----------------------------------------------------------------------------
    # Set random seed
    random_seed = 42
    np.random.seed(random_seed)
    rn.seed(random_seed)

    classifier_dir = PROJECT_ROOT / 'outputs' / 'classifier_analysis'
    classifier_dataset_dir = classifier_dir / dataset_name
    classifier_dataset_analysis_dir = classifier_dataset_dir / '{:02d}_vs_{:02d}'.format(hc_label, disease_label)

    classifier_storage_dir = classifier_dataset_analysis_dir / 'models'
    generalization_dir = classifier_dataset_analysis_dir / 'generalization'
    generalization_dir.mkdir(exist_ok=True)

    evaluated_dataset_df = load_dataset(participants_path, ids_path, freesurfer_path)

    aucs_test = []
    # ----------------------------------------------------------------------------
    for i_bootstrap in tqdm(range(n_bootstrap)):
        rvm = load(classifier_storage_dir / '{:03d}_rvr.joblib'.format(i_bootstrap))
        scaler = load(classifier_storage_dir / '{:03d}_scaler.joblib'.format(i_bootstrap))

        x_data = evaluated_dataset_df[COLUMNS_NAME].values

        tiv = evaluated_dataset_df['EstimatedTotalIntraCranialVol'].values
        tiv = tiv[:, np.newaxis]

        x_data = (np.true_divide(x_data, tiv)).astype('float32')

        x_data = np.concatenate((x_data[evaluated_dataset_df['Diagn'] == hc_label],
                                 x_data[evaluated_dataset_df['Diagn'] == disease_label]), axis=0)

        y_data = np.concatenate((np.zeros(sum(evaluated_dataset_df['Diagn'] == hc_label)),
                                 np.ones(sum(evaluated_dataset_df['Diagn'] == disease_label))))

        # Scaling using inter-quartile
        x_data = scaler.transform(x_data)

        pred = rvm.predict(x_data)
        predictions_proba = rvm.predict_proba(x_data)

        auc = roc_auc_score(y_data, predictions_proba[:, 1])

        aucs_test.append(auc)

    aucs_df = pd.DataFrame(columns=['AUCs'], data=aucs_test)
    aucs_df.to_csv(generalization_dir / '{:}_aucs.csv'.format(evaluated_dataset), index=False)

    results = pd.DataFrame(columns=['Measure', 'Value'])
    results = results.append({'Measure': 'mean', 'Value': np.mean(aucs_test)}, ignore_index=True)
    results = results.append({'Measure': 'upper_limit', 'Value': np.percentile(aucs_test, 97.5)}, ignore_index=True)
    results = results.append({'Measure': 'lower_limit', 'Value': np.percentile(aucs_test, 2.5)}, ignore_index=True)
    results.to_csv(generalization_dir / '{:}_aucs_summary.csv'.format(evaluated_dataset), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--dataset_name',
                        dest='dataset_name',
                        help='Dataset name to train the classifiers.')
    parser.add_argument('-L', '--disease_label',
                        dest='disease_label',
                        help='Disease label to train the classifiers.',
                        type=int)

    parser.add_argument('-E', '--evaluated_dataset_name',
                        dest='evaluated_dataset',
                        help='Dataset name to evaluate the classifiers.')
    args = parser.parse_args()

    main(args.dataset_name, args.disease_label, args.evaluated_dataset)
