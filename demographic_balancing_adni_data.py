#!/usr/bin/env python3
"""Script to create an homogeneous sample for the ADNI dataset.

Labels encoding
"1": "Healthy Controls",
"27": "Early mild cognitive impairment (EMCI)"
"28": "Late mild cognitive impairment (LMCI)"
"17": "Alzheimer's Disease",

excluded from study
"18": "Mild Cognitive Impairment" (excluded to simplify analysis)
"26": "Significant Memory Concern (SMC)"
"""
from pathlib import Path
import math

import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind, f_oneway

from utils import load_dataset

PROJECT_ROOT = Path.cwd()


def main():
    """Verify age and gender balance along the groups from the ADNI dataset."""
    # ----------------------------------------------------------------------------------------
    dataset_name = 'ADNI'

    participants_path = PROJECT_ROOT / 'data' / dataset_name / 'participants.tsv'
    freesurfer_path = PROJECT_ROOT / 'data' / dataset_name / 'freesurferData.csv'

    outputs_dir = PROJECT_ROOT / 'outputs'
    ids_path = outputs_dir / (dataset_name + '_cleaned_ids.csv')

    dataset_df = load_dataset(participants_path, ids_path, freesurfer_path)
    dataset_df = dataset_df[dataset_df['Diagn'].isin([1, 17, 27, 28])]
    dataset_df = dataset_df.reset_index(drop=True)
    dataset_df = dataset_df.set_index('participant_id')

    # ----------------------------------------------------------------------------------------
    print('Analysing {:}'.format(dataset_name))
    print('Total of participants = {:}'.format(len(dataset_df)))
    print('')
    print('Number of participants per diagnosis')
    print(dataset_df.groupby('Diagn')['Image_ID'].count())
    print('')

    contingency_table = pd.crosstab(dataset_df.Gender, dataset_df.Diagn)
    print('Contigency table of gender x diagnosis')
    print(contingency_table)
    print('')

    def print_age_stats(dataset_df):
        hc_age = dataset_df[dataset_df['Diagn'] == 1].Age.values
        emci_age = dataset_df[dataset_df['Diagn'] == 27].Age.values
        lmci_age = dataset_df[dataset_df['Diagn'] == 28].Age.values
        ad_age = dataset_df[dataset_df['Diagn'] == 17].Age.values

        print('Age per diagnosis')
        print('HC = {:.1f}±{:.1f} [{:d}, {:d}]'.format(hc_age.mean(), hc_age.std(),
                                                       math.ceil(hc_age.min()), math.ceil(hc_age.max())))
        print('EMCI = {:.1f}±{:.1f} [{:d}, {:d}]'.format(emci_age.mean(), emci_age.std(),
                                                         math.ceil(emci_age.min()), math.ceil(emci_age.max())))
        print('LMCI = {:.1f}±{:.1f} [{:d}, {:d}]'.format(lmci_age.mean(), lmci_age.std(),
                                                         math.ceil(lmci_age.min()), math.ceil(lmci_age.max())))
        print('AD = {:.1f}±{:.1f} [{:d}, {:d}]'.format(ad_age.mean(), ad_age.std(),
                                                       math.ceil(ad_age.min()), math.ceil(ad_age.max())))
        print('')

    print_age_stats(dataset_df)

    # ----------------------------------------------------------------------------------------
    # Gender analysis
    print('------------- GENDER ANALYSIS ----------------')

    def print_gender_analysis(contingency_table):
        _, p_value, _, _ = chi2_contingency(contingency_table[[1, 27]], correction=False)
        print('Gender - HC vs EMCI p value {:.4f}'.format(p_value))
        _, p_value, _, _ = chi2_contingency(contingency_table[[1, 28]], correction=False)
        print('Gender - HC vs LMCI p value {:.4f}'.format(p_value))
        _, p_value, _, _ = chi2_contingency(contingency_table[[1, 17]], correction=False)
        print('Gender - HC vs AD p value {:.4f}'.format(p_value))
        _, p_value, _, _ = chi2_contingency(contingency_table[[17, 27]], correction=False)
        print('Gender - AD vs EMCI p value {:.4f}'.format(p_value))
        _, p_value, _, _ = chi2_contingency(contingency_table[[17, 28]], correction=False)
        print('Gender - AD vs LMCI p value {:.4f}'.format(p_value))
        _, p_value, _, _ = chi2_contingency(contingency_table[[27, 28]], correction=False)
        print('Gender - EMCI vs LMCI p value {:.4f}'.format(p_value))
        _, p_value, _, _ = chi2_contingency(contingency_table, correction=False)
        print('Gender - TOTAL p value {:.4f}'.format(p_value))
        print('')

    print_gender_analysis(contingency_table)

    # HC have too many women
    # Removing oldest women to help balancing age
    dataset_corrected_df = dataset_df

    for _ in range(54):
        conditional_mask = (dataset_corrected_df['Diagn'] == 1) & (dataset_corrected_df['Gender'] == 0)

        hc_age = dataset_corrected_df[conditional_mask].Age

        dataset_corrected_df = dataset_corrected_df.drop(hc_age.argmax(), axis=0)
        dataset_corrected_df = dataset_corrected_df.reset_index(drop=True)

        contingency_table = pd.crosstab(dataset_corrected_df.Gender, dataset_corrected_df.Diagn)
        print_gender_analysis(contingency_table)

    # ----------------------------------------------------------------------------------------
    # Age analysis
    print('------------- AGE ANALYSIS ----------------')
    print_age_stats(dataset_corrected_df)

    def print_age_analysis(dataset_df):
        hc_age = dataset_df[dataset_df['Diagn'] == 1].Age.values
        emci_age = dataset_df[dataset_df['Diagn'] == 27].Age.values
        lmci_age = dataset_df[dataset_df['Diagn'] == 28].Age.values
        ad_age = dataset_df[dataset_df['Diagn'] == 17].Age.values

        _, p_value = ttest_ind(hc_age, emci_age)
        print('Age - HC vs EMCI p value {:.4f}'.format(p_value))
        _, p_value = ttest_ind(hc_age, lmci_age)
        print('Age - HC vs LMCI p value {:.4f}'.format(p_value))
        _, p_value = ttest_ind(hc_age, ad_age)
        print('Age - HC vs AD p value {:.4f}'.format(p_value))
        _, p_value = ttest_ind(ad_age, emci_age)
        print('Age - AD vs EMCI p value {:.4f}'.format(p_value))
        _, p_value = ttest_ind(ad_age, lmci_age)
        print('Age - AD vs LMCI p value {:.4f}'.format(p_value))
        _, p_value = ttest_ind(emci_age, lmci_age)
        print('Age - EMCI vs LMCI p value {:.4f}'.format(p_value))
        print('Age - TOTAL p value {:.4f}'.format(f_oneway(hc_age, ad_age, emci_age, lmci_age).pvalue))
        print()
        print('')

    print_age_analysis(dataset_corrected_df)

    # ----------------------------------------------------------------------------------------
    # Final dataset
    print('------------- FINAL DATASET ----------------')
    print_gender_analysis(contingency_table)
    print_age_stats(dataset_corrected_df)
    print_age_analysis(dataset_corrected_df)

    dataset_corrected_df[['Image_ID']].to_csv(outputs_dir / (dataset_name + '_homogeneous_ids.csv'), index=False)


if __name__ == "__main__":
    main()
