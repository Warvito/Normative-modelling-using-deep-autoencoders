"""Script to perform the group analysis."""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
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


def gaussian_likelihood(x):
    return np.exp(norm.logpdf(x, loc=0, scale=1))

def main():
    """"""
    # ----------------------------------------------------------------------------
    n_bootstrap = 20

    experiment_name = 'biobank_scanner1'
    model_name = 'supervised_aae'
    dataset_name = 'ADNI'

    participants_path = PROJECT_ROOT / 'data' / 'datasets' / dataset_name / 'participants.tsv'
    freesurfer_path = PROJECT_ROOT / 'data' / 'datasets' / dataset_name / 'freesurferData.csv'

    hc_label = 1
    # disease_label = 17 # AD
    disease_label = 28

    # ----------------------------------------------------------------------------
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    bootstrap_dir = experiment_dir / 'bootstrap_analysis'
    model_dir = bootstrap_dir / model_name
    ids_path = PROJECT_ROOT / 'outputs' / experiment_name / (dataset_name + '_homogeneous_ids.csv')

    for i_bootstrap in range(n_bootstrap):
        bootstrap_model_dir = model_dir / '{:03d}'.format(i_bootstrap)

        output_dataset_dir = bootstrap_model_dir / dataset_name


        # ----------------------------------------------------------------------------
        clinical_df = load_dataset(participants_path, ids_path, freesurfer_path)

        # ----------------------------------------------------------------------------
        normalized_df = pd.read_csv(output_dataset_dir / 'normalized.csv')
        reconstruction_error_df = pd.read_csv(output_dataset_dir / 'reconstruction_error.csv')
        reconstruction_df = pd.read_csv(output_dataset_dir / 'reconstruction.csv')
        encoded_df = pd.read_csv(output_dataset_dir / 'encoded.csv')

        # ----------------------------------------------------------------------------
        error_hc = reconstruction_error_df.loc[clinical_df['Diagn'] == hc_label]['Reconstruction error']
        error_patient = reconstruction_error_df.loc[clinical_df['Diagn'] == disease_label]['Reconstruction error']

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        boxplot = ax.boxplot([error_hc.values, error_patient.values], notch="True", showfliers=False, patch_artist=True)

        colors = ['white', 'lightgray']
        for patch, color in zip(boxplot['boxes'], colors):
            patch.set_facecolor(color)
        ax.yaxis.grid(True)
        plt.savefig(output_dataset_dir / 'error_analysis.png')
        plt.close()
        plt.clf()

        statistic, pvalue = stats.mannwhitneyu(error_hc.values, error_patient.values)
        effect_size = cliff_delta(error_hc.values, error_patient.values)

        error_df = pd.DataFrame({'pvalue': [pvalue], 'statistic': [statistic], 'effect_size': [effect_size]})
        error_df.to_csv(output_dataset_dir / 'error_analysis.csv', index=False)

        # ----------------------------------------------------------------------------
        region_df = pd.DataFrame(columns=['regions', 'pvalue', 'effect_size'])
        for region in COLUMNS_NAME:
            x_patient = normalized_df.loc[clinical_df['Diagn'] == disease_label][region]
            x_hc = normalized_df.loc[clinical_df['Diagn'] == hc_label][region]

            recon_patient = reconstruction_df.loc[clinical_df['Diagn'] == disease_label][region]
            recon_hc = reconstruction_df.loc[clinical_df['Diagn'] == hc_label][region]

            diff_hc = np.abs(x_hc.values - recon_hc.values)
            diff_patient = np.abs(x_patient.values - recon_patient.values)

            statistic, pvalue = stats.mannwhitneyu(diff_hc, diff_patient)
            effect_size = cliff_delta(diff_hc, diff_patient)

            print('{:}:{:6.4f}'.format(region, pvalue))

            region_df = region_df.append({'regions': region, 'pvalue': pvalue, 'effect_size': effect_size},
                                         ignore_index=True)

        region_df.to_csv(output_dataset_dir / 'regions_analysis.csv', index=False)

        # ----------------------------------------------------------------------------
        fpr, tpr, threshold = roc_curve(list(np.zeros_like(error_hc)) + list(np.ones_like(error_patient)),
                                        list(error_hc) + list(error_patient))
        roc_auc = auc(fpr, tpr)

        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(output_dataset_dir / 'auc_roc.png')
        plt.close()
        plt.clf()

        # ----------------------------------------------------------------------------

        if 'aae' in model_name:
            encoded_hc = encoded_df.loc[clinical_df['Diagn'] == hc_label][encoded_df.columns[1:]]
            encoded_patient = encoded_df.loc[clinical_df['Diagn'] == disease_label][encoded_df.columns[1:]]

            encoded_hc = encoded_hc.apply(pd.to_numeric)
            encoded_patient = encoded_patient.apply(pd.to_numeric)

            likelihood_hc = np.ones((encoded_hc.shape[0], 1))
            likelihood_patient = np.ones((encoded_patient.shape[0], 1))

            for i_latent in range(encoded_hc.shape[1]):
                likelihood_hc = np.multiply(likelihood_hc, gaussian_likelihood(encoded_hc.values[:, i_latent])[:,np.newaxis])
                likelihood_patient = np.multiply(likelihood_patient, gaussian_likelihood(encoded_patient.values[:, i_latent])[:,np.newaxis])


            fig, ax = plt.subplots()
            ax.scatter(error_hc.values, likelihood_hc, color="g", marker="o", s=10, label=str('HC'))
            ax.scatter(error_patient.values, likelihood_patient, color="r", marker="v", s=10, label=str('PATIENT'))

            # Fix auto scale error https://github.com/matplotlib/matplotlib/issues/6015
            ax.plot(error_hc.values, likelihood_hc, color='none')
            ax.relim()
            ax.autoscale_view()
            plt.xlabel('Reconstruction error')
            plt.ylabel('Likelihood')
            plt.legend(loc='upper right')
            plt.savefig(output_dataset_dir / 'likelihood.png')
            plt.close()
            plt.clf()


if __name__ == "__main__":
    main()
