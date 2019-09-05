"""Script to get the classification performance."""
from pathlib import Path
import random as rn

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from utils import COLUMNS_NAME, load_dataset

PROJECT_ROOT = Path.cwd()


def main():
    # ----------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'
    dataset_name = 'FBF_Brescia'

    participants_path = PROJECT_ROOT / 'data' / 'datasets' / dataset_name / 'participants.tsv'
    freesurfer_path = PROJECT_ROOT / 'data' / 'datasets' / dataset_name / 'freesurferData.csv'

    hc_label = 1
    disease_label = 17

    # ----------------------------------------------------------------------------
    ids_path = PROJECT_ROOT / 'outputs' / experiment_name / (dataset_name + '_homogeneous_ids.csv')

    classifier_dir = PROJECT_ROOT / 'outputs' / experiment_name / 'classifier_analysis'
    classifier_dir.mkdir(exist_ok=True)
    classifier_dataset_dir = classifier_dir / dataset_name
    classifier_dataset_dir.mkdir(exist_ok=True)

    classifier_dataset_analysis_dir = classifier_dataset_dir / '{:02d}_vs_{:02d}'.format(hc_label, disease_label)
    classifier_dataset_analysis_dir.mkdir(exist_ok=True)

    # ----------------------------------------------------------------------------
    # Set random seed
    random_seed = 42
    np.random.seed(random_seed)
    rn.seed(random_seed)

    dataset_df = load_dataset(participants_path, ids_path, freesurfer_path)

    dataset_df = dataset_df.loc[(dataset_df['Diagn'] == hc_label) | (dataset_df['Diagn'] == disease_label)]
    dataset_df = dataset_df.reset_index(drop=True)

    x_data = dataset_df[COLUMNS_NAME].values

    tiv = dataset_df['EstimatedTotalIntraCranialVol'].values
    tiv = tiv[:, np.newaxis]

    x_data = (np.true_divide(x_data, tiv)).astype('float32')

    x_data = np.concatenate((x_data[dataset_df['Diagn'] == hc_label],
                             x_data[dataset_df['Diagn'] == disease_label]), axis=0)

    y_data = np.concatenate((np.zeros(sum(dataset_df['Diagn'] == hc_label)),
                             np.ones(sum(dataset_df['Diagn'] == disease_label))))

    predictions_df = pd.DataFrame(dataset_df[['Participant_ID', 'Diagn']])
    predictions_df = predictions_df.set_index('Participant_ID')

    n_repetitions = 10
    n_folds = 10
    n_nested_folds = 5

    auc_roc_df = pd.DataFrame(columns=['CV', 'auc'])
    i_iteration = 0

    for i_repetition in range(n_repetitions):
        repetition_column_name = 'Prediction repetition {:02d}'.format(i_repetition)
        predictions_df[repetition_column_name] = np.nan

        # Create 10-fold cross-validation scheme stratified by age
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=i_repetition)
        for i_fold, (train_index, test_index) in enumerate(skf.split(x_data, y_data)):

            print('Running repetition {:02d}, fold {:02d}'.format(i_repetition, i_fold))

            x_train, x_test = x_data[train_index], x_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]

            # Scaling using inter-quartile
            scaler = RobustScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            # Systematic search for best hyperparameters
            svm = SVC(kernel='linear', probability=True)

            search_space = {'C': [2 ** -7, 2 ** -5, 2 ** -3, 2 ** -1, 2 ** 0, 2 ** 1, 2 ** 3, 2 ** 5, 2 ** 7]}

            nested_skf = StratifiedKFold(n_splits=n_nested_folds, shuffle=True, random_state=i_repetition)

            gridsearch = GridSearchCV(svm,
                                      param_grid=search_space,
                                      scoring='neg_mean_absolute_error',
                                      refit=True, cv=nested_skf,
                                      verbose=3, n_jobs=1)

            gridsearch.fit(x_train, y_train)

            best_svm = gridsearch.best_estimator_

            predictions = best_svm.predict_proba(x_test)


            auc = roc_auc_score(y_test, predictions[:,1])
            auc_roc_df = auc_roc_df.append({'CV': i_iteration, 'auc': auc}, ignore_index=True)

            i_iteration = i_iteration + 1

    auc_roc_df.to_csv(classifier_dataset_analysis_dir / 'auc.csv', index=False)
    print('{:.03f}+- {:.03f}'.format(np.mean(auc_roc_df['auc'].values), 1.96*np.std(auc_roc_df['auc'].values)))


if __name__ == "__main__":
    main()
