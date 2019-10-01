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
    dataset_name = 'PPMI'

    participants_path = PROJECT_ROOT / 'data' / 'datasets' / dataset_name / 'participants.tsv'
    freesurfer_path = PROJECT_ROOT / 'data' / 'datasets' / dataset_name / 'freesurferData.csv'

    hc_label = 1
    disease_label = 24

    # ----------------------------------------------------------------------------
    # Set random seed
    random_seed = 42
    np.random.seed(random_seed)
    rn.seed(random_seed)

    classifier_dir = PROJECT_ROOT / 'outputs' / experiment_name / 'classifier_analysis'
    classifier_dataset_dir = classifier_dir / dataset_name
    classifier_dataset_analysis_dir = classifier_dataset_dir / '{:02d}_vs_{:02d}'.format(hc_label, disease_label)
    ids_dir = classifier_dataset_analysis_dir / 'ids'
    auc_bootstrap = []
    # ----------------------------------------------------------------------------
    for i_bootstrap in range(n_bootstrap):
        # Salvar apenas as aucs
        ids_filename = 'homogeneous_bootstrap_{:03d}.csv'.format(i_bootstrap)
        ids_path = ids_dir / ids_filename

        dataset_df = load_dataset(participants_path, ids_path, freesurfer_path)

        x_data = dataset_df[COLUMNS_NAME].values

        tiv = dataset_df['EstimatedTotalIntraCranialVol'].values
        tiv = tiv[:, np.newaxis]

        x_data = (np.true_divide(x_data, tiv)).astype('float32')

        x_data = np.concatenate((x_data[dataset_df['Diagn'] == hc_label],
                                 x_data[dataset_df['Diagn'] == disease_label]), axis=0)

        y_data = np.concatenate((np.zeros(sum(dataset_df['Diagn'] == hc_label)),
                                 np.ones(sum(dataset_df['Diagn'] == disease_label))))

        n_folds = 10
        n_nested_folds = 5

        auc_list = []
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        for i_fold, (train_index, test_index) in enumerate(skf.split(x_data, y_data)):
            print('Running bootstrap {:02d}, fold {:02d}'.format(i_bootstrap, i_fold))

            x_train, x_test = x_data[train_index], x_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]

            # Scaling using inter-quartile
            scaler = RobustScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            # Systematic search for best hyperparameters
            svm = SVC(kernel='linear', probability=True)

            search_space = {'C': [2 ** -7, 2 ** -5, 2 ** -3, 2 ** -1, 2 ** 0, 2 ** 1, 2 ** 3, 2 ** 5, 2 ** 7]}

            nested_skf = StratifiedKFold(n_splits=n_nested_folds, shuffle=True, random_state=random_seed)

            gridsearch = GridSearchCV(svm,
                                      param_grid=search_space,
                                      scoring='roc_auc',
                                      refit=True, cv=nested_skf,
                                      verbose=3, n_jobs=1)

            gridsearch.fit(x_train, y_train)

            best_svm = gridsearch.best_estimator_

            predictions = best_svm.predict_proba(x_test)

            auc = roc_auc_score(y_test, predictions[:,1])
            auc_list.append(auc)

        auc_bootstrap.append(np.mean(np.array(auc_list)))
        print('AUC = {:.03f}'.format(np.mean(np.array(auc_list))))

    np.save(classifier_dataset_analysis_dir / 'aucs.npy',np.array(auc_bootstrap))


if __name__ == "__main__":
    main()
