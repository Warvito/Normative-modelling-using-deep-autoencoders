#!/usr/bin/env python3
"""Script to get the classification performance."""
import argparse
from pathlib import Path
import random as rn

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn_rvm.em_rvm import EMRVC
from tqdm import tqdm
from joblib import dump

from utils import COLUMNS_NAME, load_dataset

PROJECT_ROOT = Path.cwd()


def main(dataset_name, disease_label):
    """Calculate the performance of the classifier in each iteration of the bootstrap method."""
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
    ids_dir = classifier_dataset_analysis_dir / 'ids'

    classifier_storage_dir = classifier_dataset_analysis_dir / 'models'
    classifier_storage_dir.mkdir(exist_ok=True)
    predictions_dir = classifier_dataset_analysis_dir / 'predictions'
    predictions_dir.mkdir(exist_ok=True)

    auc_bootstrap_train = []
    auc_bootstrap_test = []
    # ----------------------------------------------------------------------------
    for i_bootstrap in tqdm(range(n_bootstrap)):
        ids_filename_train = 'homogeneous_bootstrap_{:03d}_train.csv'.format(i_bootstrap)
        ids_path_train = ids_dir / ids_filename_train

        dataset_df = load_dataset(participants_path, ids_path_train, freesurfer_path)

        x_data = dataset_df[COLUMNS_NAME].values

        tiv = dataset_df['EstimatedTotalIntraCranialVol'].values
        tiv = tiv[:, np.newaxis]

        x_data = (np.true_divide(x_data, tiv)).astype('float32')

        x_data = np.concatenate((x_data[dataset_df['Diagn'] == hc_label],
                                 x_data[dataset_df['Diagn'] == disease_label]), axis=0)

        y_data = np.concatenate((np.zeros(sum(dataset_df['Diagn'] == hc_label)),
                                 np.ones(sum(dataset_df['Diagn'] == disease_label))))

        # Scaling using inter-quartile
        scaler = RobustScaler()
        x_data = scaler.fit_transform(x_data)

        rvm = EMRVC(kernel='linear')
        rvm.fit(x_data, y_data)

        pred = rvm.predict(x_data)
        predictions_proba = rvm.predict_proba(x_data)

        auc = roc_auc_score(y_data, predictions_proba[:, 1])

        auc_bootstrap_train.append(auc)

        predictions_df = dataset_df[['Image_ID']].copy()
        predictions_df['predictions'] = pred

        dump(rvm, classifier_storage_dir / '{:03d}_rvr.joblib'.format(i_bootstrap))
        dump(scaler, classifier_storage_dir / '{:03d}_scaler.joblib'.format(i_bootstrap))

        # -----------------------------------------------------------------
        ids_filename_test = 'homogeneous_bootstrap_{:03d}_test.csv'.format(i_bootstrap)
        ids_path_test = ids_dir / ids_filename_test

        dataset_df = load_dataset(participants_path, ids_path_test, freesurfer_path)

        x_test = dataset_df[COLUMNS_NAME].values

        tiv = dataset_df['EstimatedTotalIntraCranialVol'].values
        tiv = tiv[:, np.newaxis]

        x_test = (np.true_divide(x_test, tiv)).astype('float32')

        x_test = np.concatenate((x_test[dataset_df['Diagn'] == hc_label],
                                 x_test[dataset_df['Diagn'] == disease_label]), axis=0)

        y_test = np.concatenate((np.zeros(sum(dataset_df['Diagn'] == hc_label)),
                                 np.ones(sum(dataset_df['Diagn'] == disease_label))))

        x_test = scaler.transform(x_test)

        pred = rvm.predict(x_test)
        predictions_proba = rvm.predict_proba(x_test)

        auc = roc_auc_score(y_test, predictions_proba[:, 1])

        auc_bootstrap_test.append(auc)

        temp_df = dataset_df[['Image_ID']].copy()
        temp_df['predictions'] = pred

        predictions_df = pd.concat([predictions_df, temp_df], axis=0)
        predictions_df.to_csv(predictions_dir / 'homogeneous_bootstrap_{:03d}_prediction.csv'.format(i_bootstrap),
                              index=False)

    np.save(classifier_dataset_analysis_dir / 'aucs_train.npy', np.array(auc_bootstrap_train))
    np.save(classifier_dataset_analysis_dir / 'aucs_test.npy', np.array(auc_bootstrap_test))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--dataset_name',
                        dest='dataset_name',
                        help='Dataset name to train the classifiers.')
    parser.add_argument('-L', '--disease_label',
                        dest='disease_label',
                        help='Disease label to train the classifiers.',
                        type=int)
    args = parser.parse_args()

    main(args.dataset_name, args.disease_label)
