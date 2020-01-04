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
from scipy import stats
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

from utils import COLUMNS_NAME, load_dataset, cliff_delta

PROJECT_ROOT = Path.cwd()


def compute_brain_regions_deviations(diff_df, clinical_df, disease_label, hc_label=1):
    """ Calculate the Cliff's delta effect size between groups."""
    region_df = pd.DataFrame(columns=['regions', 'pvalue', 'effect_size'])

    diff_hc = diff_df.loc[clinical_df['Diagn'] == disease_label]
    diff_patient = diff_df.loc[clinical_df['Diagn'] == hc_label]

    for region in COLUMNS_NAME:
        _, pvalue = stats.mannwhitneyu(diff_hc[region], diff_patient[region])
        effect_size = cliff_delta(diff_hc[region].values, diff_patient[region].values)

        region_df = region_df.append({'regions': region, 'pvalue': pvalue, 'effect_size': effect_size},
                                     ignore_index=True)

    return region_df


def compute_classification_performance(reconstruction_error_df, clinical_df, disease_label, hc_label=1):
    """ Calculate the AUCs of the normative model."""
    error_hc = reconstruction_error_df.loc[clinical_df['Diagn'] == hc_label]['Reconstruction error']
    error_patient = reconstruction_error_df.loc[clinical_df['Diagn'] == disease_label]['Reconstruction error']

    fpr, tpr, _ = roc_curve(list(np.zeros_like(error_hc)) + list(np.ones_like(error_patient)),
                            list(error_hc) + list(error_patient))

    roc_auc = auc(fpr, tpr)

    tpr = np.interp(np.linspace(0, 1, 101), fpr, tpr)

    tpr[0] = 0.0

    return roc_auc, tpr


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

    # ----------------------------------------------------------------------------
    clinical_df = load_dataset(participants_path, ids_path, freesurfer_path)
    clinical_df = clinical_df.set_index('participant_id')

    tpr_list = []
    auc_roc_list = []
    effect_size_list = []

    for i_bootstrap in tqdm(range(n_bootstrap)):
        bootstrap_model_dir = model_dir / '{:03d}'.format(i_bootstrap)

        output_dataset_dir = bootstrap_model_dir / dataset_name
        output_dataset_dir.mkdir(exist_ok=True)

        analysis_dir = output_dataset_dir / '{:02d}_vs_{:02d}'.format(hc_label, disease_label)
        analysis_dir.mkdir(exist_ok=True)

        # ----------------------------------------------------------------------------
        normalized_df = pd.read_csv(output_dataset_dir / 'normalized.csv', index_col='participant_id')
        reconstruction_df = pd.read_csv(output_dataset_dir / 'reconstruction.csv', index_col='participant_id')
        reconstruction_error_df = pd.read_csv(output_dataset_dir / 'reconstruction_error.csv',
                                              index_col='participant_id')

        # ----------------------------------------------------------------------------
        # Compute effect size of the brain regions for the bootstrap iteration
        diff_df = np.abs(normalized_df - reconstruction_df)
        region_df = compute_brain_regions_deviations(diff_df, clinical_df, disease_label)
        effect_size_list.append(region_df['effect_size'].values)
        region_df.to_csv(analysis_dir / 'regions_analysis.csv', index=False)

        # ----------------------------------------------------------------------------
        # Compute AUC-ROC for the bootstrap iteration
        roc_auc, tpr = compute_classification_performance(reconstruction_error_df, clinical_df, disease_label)
        auc_roc_list.append(roc_auc)
        tpr_list.append(tpr)

    (bootstrap_dir / dataset_name).mkdir(exist_ok=True)
    comparison_dir = bootstrap_dir / dataset_name / ('{:02d}_vs_{:02d}'.format(hc_label, disease_label))
    comparison_dir.mkdir(exist_ok=True)

    # ----------------------------------------------------------------------------
    # Save regions effect sizes
    effect_size_df = pd.DataFrame(columns=COLUMNS_NAME, data=np.array(effect_size_list))
    effect_size_df.to_csv(comparison_dir / 'effect_size.csv')

    # Save AUC bootstrap values
    auc_roc_list = np.array(auc_roc_list)
    auc_roc_df = pd.DataFrame(columns=['AUC-ROC'], data=auc_roc_list)
    auc_roc_df.to_csv(comparison_dir / 'auc_rocs.csv', index=False)

    # ----------------------------------------------------------------------------
    # Create Figure 3 of the paper
    tpr_list = np.array(tpr_list)
    mean_tprs = tpr_list.mean(axis=0)
    tprs_upper = np.percentile(tpr_list, 97.5, axis=0)
    tprs_lower = np.percentile(tpr_list, 2.5, axis=0)

    plt.plot(np.linspace(0, 1, 101),
             mean_tprs,
             'b', lw=2,
             label='ROC curve (AUC = {:0.3f} ; 95% CI [{:0.3f}, {:0.3f}])'.format(np.mean(auc_roc_list),
                                                                                  np.percentile(auc_roc_list, 2.5),
                                                                                  np.percentile(auc_roc_list, 97.5)))

    plt.fill_between(np.linspace(0, 1, 101),
                     tprs_lower, tprs_upper,
                     color='grey', alpha=0.2)

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
    # Create figure for supplementary materials
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

    # --------------------------------------------------------------------------------------------
    # Create Figure 4 of the paper
    effect_size_sig_df = effect_size_df.reindex(effect_size_df.mean().sort_values().index, axis=1)
    lower_bound = np.percentile(effect_size_sig_df, 2.5, axis=0)
    higher_bound = np.percentile(effect_size_sig_df, 97.5, axis=0)

    for i, column in enumerate(effect_size_sig_df.columns):
        if (lower_bound[i] < 0) & (higher_bound[i] > 0):
            effect_size_sig_df = effect_size_sig_df.drop(columns=column)

    n_regions = len(effect_size_sig_df.columns)

    plt.figure()
    plt.hlines(range(n_regions),
               np.percentile(effect_size_sig_df, 2.5, axis=0),
               np.percentile(effect_size_sig_df, 97.5, axis=0))

    plt.plot(effect_size_sig_df.mean().values, range(n_regions), 's', color='k')
    plt.axvline(0, ls='--')
    plt.yticks(np.arange(n_regions), effect_size_sig_df.columns)
    plt.xlabel('Effect size')
    plt.ylabel('Brain regions')
    plt.tight_layout()
    plt.savefig(comparison_dir / 'Significant_regions.eps', format='eps')
    plt.close()
    plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--dataset_name',
                        dest='dataset_name',
                        help='Dataset name to perform group analysis.')
    parser.add_argument('-L', '--disease_label',
                        dest='disease_label',
                        help='Disease label to perform group analysis.',
                        type=int)
    args = parser.parse_args()

    main(args.dataset_name, args.disease_label)
