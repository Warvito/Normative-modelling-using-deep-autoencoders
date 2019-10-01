"""Deterministic supervised adversarial autoencoder."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_curve, auc

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
    """"""
    # ----------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'
    model_name = 'supervised_aae'
    dataset_name = 'ADNI'

    participants_path = PROJECT_ROOT / 'data' / 'datasets' / dataset_name / 'participants.tsv'
    freesurfer_path = PROJECT_ROOT / 'data' / 'datasets' / dataset_name / 'freesurferData.csv'

    hc_label = 1
    disease_label = 27

    # ----------------------------------------------------------------------------
    # Create directories structure
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    individual_dir = experiment_dir / 'individual_analysis'
    individual_dir.mkdir(exist_ok=True)

    model_dir = individual_dir / model_name
    model_dir.mkdir(exist_ok=True)

    output_dataset_dir = model_dir / dataset_name
    output_dataset_dir.mkdir(exist_ok=True)

    # ----------------------------------------------------------------------------
    # Loading data
    n_repetitions = 10
    n_folds = 5

    tprs = []
    base_fpr = np.linspace(0, 1, 101)

    auc_roc_list = []
    effect_size_list = []

    for i_repetition in range(n_repetitions):
        for i_fold in range(n_folds):
            print('Running repetition {:02d}, fold {:02d}'.format(i_repetition, i_fold))

            repetition_dir = output_dataset_dir / '{}_{}'.format(i_repetition, i_fold)

            ids_path = repetition_dir / 'test_ids.csv'
            clinical_df = load_dataset(participants_path, ids_path, freesurfer_path)
            total_reconstruction_error_df = pd.read_csv(repetition_dir / 'reconstruction_error.csv')

            reconstruction_error_df = pd.merge(clinical_df, total_reconstruction_error_df, on='Participant_ID', how='inner')

            error_hc = reconstruction_error_df.loc[clinical_df['Diagn'] == hc_label]['Reconstruction error']
            error_patient = reconstruction_error_df.loc[clinical_df['Diagn'] == disease_label]['Reconstruction error']

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            boxplot = ax.boxplot([error_hc.values, error_patient.values], notch="True", showfliers=False, patch_artist=True)

            colors = ['white', 'lightgray']
            for patch, color in zip(boxplot['boxes'], colors):
                patch.set_facecolor(color)
            ax.yaxis.grid(True)
            plt.savefig(repetition_dir / 'error_analysis.png')
            plt.close()
            plt.clf()

            statistic, pvalue = stats.mannwhitneyu(error_hc.values, error_patient.values)
            effect_size = cliff_delta(error_hc.values, error_patient.values)

            error_df = pd.DataFrame({'pvalue': [pvalue], 'statistic': [statistic], 'effect_size': [effect_size]})
            error_df.to_csv(repetition_dir / 'error_analysis.csv', index=False)

            # ----------------------------------------------------------------------------
            fpr, tpr, threshold = roc_curve(list(np.zeros_like(error_hc)) + list(np.ones_like(error_patient)),
                                            list(error_hc) + list(error_patient))
            roc_auc = auc(fpr, tpr)
            auc_roc_list.append(roc_auc)

            plt.title('Receiver Operating Characteristic')
            plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.savefig(repetition_dir / 'auc_roc.png')
            plt.close()
            plt.clf()

            tpr = np.interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs.append(tpr)


    (output_dataset_dir / ('{:02d}_vs_{:02d}'.format(hc_label, disease_label))).mkdir(exist_ok=True)

    auc_roc_list = np.array(auc_roc_list)

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)

    tprs_upper = np.percentile(tprs, 97.5, axis=0)
    tprs_lower = np.percentile(tprs, 2.5, axis=0)

    plt.plot(base_fpr, mean_tprs, 'b',lw=2, label='ROC curve (AUC = {:0.3f} ; 95% CI [{:0.3f}, {:0.3f}])'.format(np.mean(auc_roc_list),
                                                                                                    np.percentile(auc_roc_list, 2.5),
                                                                                                    np.percentile(auc_roc_list, 97.5)))

    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2)

    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(output_dataset_dir / ('{:02d}_vs_{:02d}'.format(hc_label, disease_label)) / 'AUC-ROC.eps',
                format='eps')
    plt.close()
    plt.clf()



if __name__ == "__main__":
    main()
