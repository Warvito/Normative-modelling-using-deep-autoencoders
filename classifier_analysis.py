"""Script to get the classification performance."""
from pathlib import Path
import random as rn

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from utils import COLUMNS_NAME, load_dataset

PROJECT_ROOT = Path.cwd()


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
    ids_dir = classifier_dataset_analysis_dir / 'ids'
    auc_bootstrap_train = []
    auc_bootstrap_test = []
    # ----------------------------------------------------------------------------
    for i_bootstrap in range(n_bootstrap):
        # Salvar apenas as aucs
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

        print('Running bootstrap {:02d}'.format(i_bootstrap))

        # Scaling using inter-quartile
        scaler = RobustScaler()
        x_data = scaler.fit_transform(x_data)

        # Systematic search for best hyperparameters
        svm = SVC(kernel='linear', probability=True)

        search_space = {'C': [2 ** -7, 2 ** -5, 2 ** -3, 2 ** -1, 2 ** 0, 2 ** 1, 2 ** 3, 2 ** 5, 2 ** 7]}
        n_nested_folds = 5
        nested_skf = StratifiedKFold(n_splits=n_nested_folds, shuffle=True, random_state=random_seed)

        gridsearch = GridSearchCV(svm,
                                  param_grid=search_space,
                                  scoring='roc_auc',
                                  refit=True, cv=nested_skf,
                                  verbose=3, n_jobs=1)

        gridsearch.fit(x_data, y_data)

        best_svm = gridsearch.best_estimator_

        predictions = best_svm.predict_proba(x_data)

        auc = roc_auc_score(y_data, predictions[:, 1])

        auc_bootstrap_train.append(auc)

        print('AUC = {:.03f}'.format(auc))

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

        predictions = best_svm.predict_proba(x_test)

        auc = roc_auc_score(y_test, predictions[:, 1])

        auc_bootstrap_test.append(auc)

        print('AUC = {:.03f}'.format(auc))

    np.save(classifier_dataset_analysis_dir / 'aucs_train.npy', np.array(auc_bootstrap_train))
    np.save(classifier_dataset_analysis_dir / 'aucs_test.npy', np.array(auc_bootstrap_test))


if __name__ == "__main__":
    main()
