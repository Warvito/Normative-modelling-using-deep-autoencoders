#!/usr/bin/env python3
""" Script to compare the performance between classifiers and normative."""
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path.cwd()


def main(training_dataset_name, testing_dataset_name, disease_label):
    hc_label = 1

    bootstrap_dir = PROJECT_ROOT / 'outputs' / 'bootstrap_analysis'
    comparison_dir = bootstrap_dir / testing_dataset_name / ('{:02d}_vs_{:02d}'.format(hc_label, disease_label))
    normative_auc_roc_df = pd.read_csv(comparison_dir / 'auc_rocs.csv')
    normative_auc_roc_df = normative_auc_roc_df.rename(columns={'AUC-ROC': 'AUCs'})

    classifier_dir = PROJECT_ROOT / 'outputs' / 'classifier_analysis' / training_dataset_name
    generalization_dir = classifier_dir / '{:02d}_vs_{:02d}'.format(hc_label, disease_label) / 'generalization'

    classifier_results = pd.read_csv(generalization_dir / '{:}_aucs.csv'.format(testing_dataset_name))

    print(np.percentile(normative_auc_roc_df - classifier_results, 2.5))
    print(np.percentile(normative_auc_roc_df - classifier_results, 97.5))

    results = pd.DataFrame(columns={'Measure', 'Value'})
    results = results.append({'Measure': 'Lower',
                              'Value': np.percentile(normative_auc_roc_df - classifier_results, 2.5)},
                             ignore_index=True)
    results = results.append({'Measure': 'Upper',
                              'Value': np.percentile(normative_auc_roc_df - classifier_results, 97.5)},
                             ignore_index=True)

    results.to_csv(generalization_dir / 'normative_vs_classifier_{}_{}.csv'.format(testing_dataset_name, disease_label), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--training_dataset_name',
                        dest='training_dataset_name',
                        help='Dataset name to perform comparison between normative and classifiers.')
    parser.add_argument('-E', '--testing_dataset_name',
                        dest='testing_dataset_name',
                        help='Dataset name to perform comparison between normative and classifiers.')
    parser.add_argument('-L', '--disease_label',
                        dest='disease_label',
                        help='Disease label to perform comparison between normative and classifiers.',
                        type=int)
    args = parser.parse_args()

    main(args.training_dataset_name, args.testing_dataset_name, args.disease_label)
