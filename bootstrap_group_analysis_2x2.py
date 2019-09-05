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
    n_bootstrap = 100

    experiment_name = 'biobank_scanner1'
    model_name = 'supervised_aae'
    dataset_name = 'ADNI'

    participants_path = PROJECT_ROOT / 'data' / 'datasets' / dataset_name / 'participants.tsv'
    freesurfer_path = PROJECT_ROOT / 'data' / 'datasets' / dataset_name / 'freesurferData.csv'

    hc_label = 1
    # disease_label = 17 # AD
    disease_label = 17

    # ----------------------------------------------------------------------------
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    bootstrap_dir = experiment_dir / 'bootstrap_analysis'
    model_dir = bootstrap_dir / model_name
    ids_path = PROJECT_ROOT / 'outputs' / experiment_name / (dataset_name + '_homogeneous_ids.csv')

    tprs = []
    base_fpr = np.linspace(0, 1, 101)

    auc_roc_list = []
    effect_size_list = []

    for i_bootstrap in range(n_bootstrap):
        bootstrap_model_dir = model_dir / '{:03d}'.format(i_bootstrap)

        output_dataset_dir = bootstrap_model_dir / dataset_name
        output_dataset_dir.mkdir(exist_ok=True)

        analysis_dir = output_dataset_dir / '{:02d}_vs_{:02d}'.format(hc_label, disease_label)
        analysis_dir.mkdir(exist_ok=True)

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
        plt.savefig(analysis_dir / 'error_analysis.png')
        plt.close()
        plt.clf()

        statistic, pvalue = stats.mannwhitneyu(error_hc.values, error_patient.values)
        effect_size = cliff_delta(error_hc.values, error_patient.values)

        error_df = pd.DataFrame({'pvalue': [pvalue], 'statistic': [statistic], 'effect_size': [effect_size]})
        error_df.to_csv(analysis_dir / 'error_analysis.csv', index=False)

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

        effect_size_list.append(region_df['effect_size'].values)

        region_df.to_csv(analysis_dir / 'regions_analysis.csv', index=False)

        # ----------------------------------------------------------------------------
        # https://stats.stackexchange.com/questions/186337/average-roc-for-repeated-10-fold-cross-validation-with-probability-estimates
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
        plt.savefig(analysis_dir / 'auc_roc.png')
        plt.close()
        plt.clf()

        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

        # ----------------------------------------------------------------------------

        if 'aae' in model_name:
            encoded = encoded_df[encoded_df.columns[1:]].apply(pd.to_numeric)
            likelihood = np.ones((encoded.shape[0], 1))

            for i_latent in range(encoded.shape[1]):
                likelihood = np.multiply(likelihood, gaussian_likelihood(encoded.values[:, i_latent])[:,np.newaxis])

            likelihood_df = pd.DataFrame(columns=['Participant_ID', 'likelihood'])
            likelihood_df['Participant_ID'] = encoded_df[encoded_df.columns[0]]
            likelihood_df['likelihood'] = likelihood

            likelihood_df.to_csv(output_dataset_dir / 'likelihood.csv', index=False)

            likelihood_hc = likelihood_df.loc[clinical_df['Diagn'] == hc_label]['likelihood']
            likelihood_patient = likelihood_df.loc[clinical_df['Diagn'] == disease_label]['likelihood']

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
            plt.savefig(analysis_dir / 'likelihood.png')
            plt.close()
            plt.clf()

    (bootstrap_dir / dataset_name).mkdir(exist_ok=True)
    (bootstrap_dir / dataset_name / ('{:02d}_vs_{:02d}'.format(hc_label, disease_label))).mkdir(exist_ok=True)


    auc_roc_list = np.array(auc_roc_list)

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    tprs_upper = np.minimum(mean_tprs + 2*std, 1)
    tprs_lower = mean_tprs - 2*std

    plt.plot(base_fpr, mean_tprs, 'b',lw=2, label='AUC = {:0.3f} +- {:0.3f}'.format(np.mean(auc_roc_list), 2*np.std(auc_roc_list)))
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(bootstrap_dir / dataset_name / ('{:02d}_vs_{:02d}'.format(hc_label, disease_label)) / 'AUC-ROC.png')
    plt.close()
    plt.clf()

    auc_roc_df = pd.DataFrame(columns=['AUC-ROC'], data=auc_roc_list)
    auc_roc_df.to_csv(bootstrap_dir / dataset_name / ('{:02d}_vs_{:02d}'.format(hc_label, disease_label)) / 'auc_rocs.csv', index=False)

    effect_size_df = pd.DataFrame(columns=COLUMNS_NAME, data=np.array(effect_size_list))
    effect_size_df.to_csv(bootstrap_dir / dataset_name / ('{:02d}_vs_{:02d}'.format(hc_label, disease_label)) / 'effect_size.csv')

    effect_size_df = effect_size_df.reindex(effect_size_df.quantile(0.5).sort_values().index, axis=1)

    plt.figure(figsize=(16, 20))
    plt.hlines(range(101), effect_size_df.quantile([.25]).values[0], effect_size_df.quantile([.75]).values[0])
    plt.plot(effect_size_df.quantile([.5]).values[0], range(101), 'o')
    plt.axvline(0, ls='--')
    plt.yticks(np.arange(101), effect_size_df.columns)
    plt.tight_layout()
    plt.savefig(bootstrap_dir / dataset_name / ('{:02d}_vs_{:02d}'.format(hc_label, disease_label)) / 'Regions.png')
    plt.close()
    plt.clf()

if __name__ == "__main__":
    main()
