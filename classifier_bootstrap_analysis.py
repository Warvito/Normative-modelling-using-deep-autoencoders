"""
Script to get the classification performance.
https://stats.stackexchange.com/questions/96739/what-is-the-632-rule-in-bootstrapping
https://github.com/rasbt/mlxtend/blob/9c044a920c31054fa106fb028e9115a3bd852cf8/mlxtend/evaluate/bootstrap_point632.py
"""
from pathlib import Path
import random as rn

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from utils import load_dataset

PROJECT_ROOT = Path.cwd()


def no_information_rate(targets, predictions, loss_fn):
    combinations = np.array(np.meshgrid(targets, predictions)).reshape(-1, 2)
    return loss_fn(combinations[:, 0], combinations[:, 1])


def main():
    # ----------------------------------------------------------------------------
    n_bootstrap = 1000

    experiment_name = 'biobank_scanner1'
    dataset_name = 'ADNI'

    participants_path = PROJECT_ROOT / 'data' / 'datasets' / dataset_name / 'participants.tsv'
    freesurfer_path = PROJECT_ROOT / 'data' / 'datasets' / dataset_name / 'freesurferData.csv'

    hc_label = 1
    disease_label = 17

    # ----------------------------------------------------------------------------
    # Set random seed
    random_seed = 42
    np.random.seed(random_seed)
    rn.seed(random_seed)

    classifier_dir = PROJECT_ROOT / 'outputs' / experiment_name / 'classifier_analysis'
    classifier_dataset_dir = classifier_dir / dataset_name
    classifier_dataset_analysis_dir = classifier_dataset_dir / '{:02d}_vs_{:02d}'.format(hc_label, disease_label)

    ids_path = PROJECT_ROOT / 'outputs' / experiment_name / (dataset_name + '_homogeneous_ids.csv')


    auc_bootstrap_train = np.load(classifier_dataset_analysis_dir / 'aucs_train.npy')
    auc_bootstrap_test = np.load(classifier_dataset_analysis_dir / 'aucs_test.npy')

    # ----------------------------------------------------------------------------
    bootstrap = []
    for i_bootstrap in range(n_bootstrap):
        print(i_bootstrap)

        predictions = pd.read_csv(classifier_dataset_analysis_dir / 'predictions'/ 'homogeneous_bootstrap_{:03d}_prediction.csv'.format(i_bootstrap))
        dataset_df = load_dataset(participants_path, ids_path, freesurfer_path)
        dataset = pd.merge(predictions, dataset_df, on='Image_ID')
        dataset['Diagn'] = dataset['Diagn'].map({hc_label: 0, disease_label: 1})


        auc_resubstitution = auc_bootstrap_train[i_bootstrap]
        auc_out_of_bag =  auc_bootstrap_test[i_bootstrap]
        gamma = no_information_rate(dataset['Diagn'].values,
                                    dataset['predictions'].values,
                                    roc_auc_score)
        R = (- (auc_out_of_bag - auc_resubstitution)) / (gamma - (1-auc_out_of_bag))
        w = 0.632/(1-0.368*R)

        bootstrap.append((w * auc_out_of_bag + (1-w)*auc_resubstitution))

    final_value = np.mean(bootstrap)
    print(final_value)
    print(np.percentile(bootstrap, 97.5))
    print(np.percentile(bootstrap, 2.5))

if __name__ == "__main__":
    main()
