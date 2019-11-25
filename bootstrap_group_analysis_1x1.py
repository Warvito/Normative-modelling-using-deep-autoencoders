#!/usr/bin/env python3
"""Script to perform the group analysis.

Creates the figures 3 and 4 from the paper

References:
    https://towardsdatascience.com/an-introduction-to-the-bootstrap-method-58bcb51b4d60
    https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/
    https://stats.stackexchange.com/questions/186337/average-roc-for-repeated-10-fold-cross-validation-with-probability-estimates
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy import stats
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

from utils import COLUMNS_NAME, load_dataset, cliff_delta

PROJECT_ROOT = Path.cwd()


def gaussian_likelihood(x):
    """Calculate the likelihood a using the Gaussian distribution."""
    return np.exp(norm.logpdf(x, loc=0, scale=1))


def main(dataset_name, disease_label):
    """Perform the group analysis."""
    # ----------------------------------------------------------------------------
    n_bootstrap = 1000

    model_name = 'supervised_aae'

    participants_path = PROJECT_ROOT / 'data' / dataset_name / 'participants.tsv'
    freesurfer_path = PROJECT_ROOT / 'data' / dataset_name / 'freesurferData.csv'

    hc_label = 1

    # ----------------------------------------------------------------------------
    bootstrap_dir = PROJECT_ROOT / 'outputs' / 'bootstrap_analysis'
    model_dir = bootstrap_dir / model_name
    ids_path = PROJECT_ROOT / 'outputs' / (dataset_name + '_homogeneous_ids.csv')

    tprs = []
    base_fpr = np.linspace(0, 1, 101)

    auc_roc_list = []
    effect_size_list = []

    for i_bootstrap in tqdm(range(n_bootstrap)):
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

        # ----------------------------------------------------------------------------
        error_hc = reconstruction_error_df.loc[clinical_df['Diagn'] == hc_label]['Reconstruction error']
        error_patient = reconstruction_error_df.loc[clinical_df['Diagn'] == disease_label]['Reconstruction error']

        # ----------------------------------------------------------------------------
        # Compute effect size of the brain regions for the bootstrap iteration
        region_df = pd.DataFrame(columns=['regions', 'pvalue', 'effect_size'])
        for region in COLUMNS_NAME:
            x_patient = normalized_df.loc[clinical_df['Diagn'] == disease_label][region]
            x_hc = normalized_df.loc[clinical_df['Diagn'] == hc_label][region]

            recon_patient = reconstruction_df.loc[clinical_df['Diagn'] == disease_label][region]
            recon_hc = reconstruction_df.loc[clinical_df['Diagn'] == hc_label][region]

            diff_hc = np.abs(x_hc.values - recon_hc.values)
            diff_patient = np.abs(x_patient.values - recon_patient.values)

            _, pvalue = stats.mannwhitneyu(diff_hc, diff_patient)
            effect_size = cliff_delta(diff_hc, diff_patient)

            # print('{:}:{:6.4f}'.format(region, pvalue))

            region_df = region_df.append({'regions': region, 'pvalue': pvalue, 'effect_size': effect_size},
                                         ignore_index=True)

        effect_size_list.append(region_df['effect_size'].values)

        region_df.to_csv(analysis_dir / 'regions_analysis.csv', index=False)

        # ----------------------------------------------------------------------------
        # Compute AUC-ROC for the bootstrap iteration
        fpr, tpr, _ = roc_curve(list(np.zeros_like(error_hc)) + list(np.ones_like(error_patient)),
                                list(error_hc) + list(error_patient))

        roc_auc = auc(fpr, tpr)
        auc_roc_list.append(roc_auc)

        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

    (bootstrap_dir / dataset_name).mkdir(exist_ok=True)
    comparison_dir = bootstrap_dir / dataset_name / ('{:02d}_vs_{:02d}'.format(hc_label, disease_label))
    comparison_dir.mkdir(exist_ok=True)

    # Save regions effect sizes
    effect_size_df = pd.DataFrame(columns=COLUMNS_NAME, data=np.array(effect_size_list))
    effect_size_df.to_csv(comparison_dir / 'effect_size.csv')

    # Save AUC bootstrap values
    auc_roc_list = np.array(auc_roc_list)
    auc_roc_df = pd.DataFrame(columns=['AUC-ROC'], data=auc_roc_list)
    auc_roc_df.to_csv(comparison_dir / 'auc_rocs.csv', index=False)

    tprs = np.array(tprs)

    # ----------------------------------------------------------------------------
    # Create Figure 3 of the paper
    mean_tprs = tprs.mean(axis=0)
    tprs_upper = np.percentile(tprs, 97.5, axis=0)
    tprs_lower = np.percentile(tprs, 2.5, axis=0)

    plt.plot(base_fpr, mean_tprs, 'b', lw=2,
             label='ROC curve (AUC = {:0.3f} ; 95% CI [{:0.3f}, {:0.3f}])'.format(np.mean(auc_roc_list),
                                                                                  np.percentile(auc_roc_list, 2.5),
                                                                                  np.percentile(auc_roc_list, 97.5)))

    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2)

    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(comparison_dir / 'AUC-ROC.eps', format='eps')
    plt.close()
    plt.clf()

    # --------------------------------------------------------------------------------------------
    # Create Figure 4 of the paper
    effect_size_df = effect_size_df.reindex(effect_size_df.mean().sort_values().index, axis=1)

    plt.figure(figsize=(16, 20))
    plt.hlines(range(101),
               np.percentile(effect_size_df, 2.5, axis=0),
               np.percentile(effect_size_df, 97.5, axis=0))

    plt.plot(effect_size_df.mean().values, range(101), 's', color='k')
    plt.axvline(0, ls='--')
    plt.yticks(np.arange(101), effect_size_df.columns)
    plt.xlabel('Effect size')
    plt.ylabel('Brain regions')
    plt.tight_layout()
    plt.savefig(comparison_dir / 'Regions.eps', format='eps')
    plt.close()
    plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--dataset_name',
                        dest='dataset_name',
                        help='Dataset name to perform group analsysis.')
    parser.add_argument('-L', '--disease_label',
                        dest='disease_label',
                        help='Disease label to perform group analsysis.',
                        type=int)
    args = parser.parse_args()

    main(args.dataset_name, args.disease_label)
