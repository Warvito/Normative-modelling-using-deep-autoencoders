#!/usr/bin/env python3
"""Script to create homogeneous samples for the AIBL dataset.

Labels encoding
"1": "Healthy Controls",
"17": "Alzheimer's Disease",
"18": "Mild Cognitive Impairment",
"""
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind, f_oneway

from utils import load_dataset

PROJECT_ROOT = Path.cwd()


def main():
    """Verify age and gender balance along the groups from the TOMC dataset."""
    # ----------------------------------------------------------------------------------------
    dataset_name = 'AIBL'

    participants_path = PROJECT_ROOT / 'data' / dataset_name / 'participants.tsv'
    freesurfer_path = PROJECT_ROOT / 'data' / dataset_name / 'freesurferData.csv'

    # ----------------------------------------------------------------------------------------
    outputs_dir = PROJECT_ROOT / 'outputs'
    ids_path = outputs_dir / (dataset_name + '_cleaned_ids.csv')

    dataset_df = load_dataset(participants_path, ids_path, freesurfer_path)
    print(dataset_df.groupby('Diagn').count())

    contingency_table = pd.crosstab(dataset_df.Gender, dataset_df.Diagn)

    _, p_value, _, _ = chi2_contingency(contingency_table[[1, 17]], correction=False)
    print('Gender - HC vs AD p value {}'.format(p_value))
    _, p_value, _, _ = chi2_contingency(contingency_table[[1, 18]], correction=False)
    print('Gender - HC vs MCI p value {}'.format(p_value))
    _, p_value, _, _ = chi2_contingency(contingency_table[[17, 18]], correction=False)
    print('Gender - MCI vs AD p value {}'.format(p_value))

    # HC have too many women
    # Remove oldest ones to help to balance age
    hc_age = dataset_df[dataset_df['Diagn'] == 1].Age.values

    index_to_remove = dataset_df[(dataset_df['Diagn'] == 1) & (dataset_df['Gender'] == 0)].iloc[hc_age.argmax()].name
    dataset_corrected_df = dataset_df.drop(index_to_remove, axis=0)
    dataset_corrected_df = dataset_corrected_df.reset_index(drop=True)

    for _ in range(25):
        hc_age = dataset_corrected_df[dataset_corrected_df['Diagn'] == 1].Age.values

        contingency_table = pd.crosstab(dataset_corrected_df.Gender, dataset_corrected_df.Diagn)

        _, p_value, _, _ = chi2_contingency(contingency_table[[1, 17]], correction=False)
        print('Gender - HC vs AD p value {}'.format(p_value))
        _, p_value, _, _ = chi2_contingency(contingency_table[[1, 18]], correction=False)
        print('Gender - HC vs MCI p value {}'.format(p_value))

        index_to_remove = \
        dataset_corrected_df[(dataset_corrected_df['Diagn'] == 1) & (dataset_corrected_df['Gender'] == 0)].iloc[
            hc_age.argmax()].name
        dataset_corrected_df = dataset_corrected_df.drop(index_to_remove, axis=0)
        dataset_corrected_df = dataset_corrected_df.reset_index(drop=True)

    hc_age = dataset_corrected_df[dataset_corrected_df['Diagn'] == 1].Age.values
    ad_age = dataset_corrected_df[dataset_corrected_df['Diagn'] == 17].Age.values
    mci_age = dataset_corrected_df[dataset_corrected_df['Diagn'] == 18].Age.values

    print(hc_age.mean())
    print(ad_age.mean())
    print(mci_age.mean())

    _, p_value = ttest_ind(hc_age, ad_age)
    print('Age - HC vs AD p value {}'.format(p_value))
    _, p_value = ttest_ind(hc_age, mci_age)
    print('Age - HC vs MCI p value {}'.format(p_value))
    _, p_value = ttest_ind(ad_age, mci_age)
    print('Age - MCI vs AD p value {}'.format(p_value))

    # hc is too old, droping some of the youngest
    dataset_corrected_df = dataset_corrected_df.drop(
        dataset_corrected_df[dataset_corrected_df['Diagn'] == 1].iloc[hc_age.argmax()].name, axis=0)
    dataset_corrected_df = dataset_corrected_df.reset_index(drop=True)
    hc_age = dataset_corrected_df[dataset_corrected_df['Diagn'] == 1].Age.values

    for _ in range(5):
        hc_age = dataset_corrected_df[dataset_corrected_df['Diagn'] == 1].Age.values
        print(hc_age.argmin())
        print(dataset_corrected_df[dataset_corrected_df['Diagn'] == 1].iloc[hc_age.argmax()].Age)
        print(dataset_corrected_df[dataset_corrected_df['Diagn'] == 1].iloc[hc_age.argmax()].name)
        print(hc_age)
        print('')

        _, p_value = ttest_ind(hc_age, ad_age)
        print('Age - HC vs AD p value {}'.format(p_value))
        _, p_value = ttest_ind(hc_age, mci_age)
        print('Age - HC vs MCI p value {}'.format(p_value))

        dataset_corrected_df = dataset_corrected_df.drop(
            dataset_corrected_df[dataset_corrected_df['Diagn'] == 1].iloc[hc_age.argmax()].name, axis=0)
        dataset_corrected_df = dataset_corrected_df.reset_index(drop=True)

    _, p_value = ttest_ind(hc_age, ad_age)
    print('Age - HC vs AD p value {}'.format(p_value))
    _, p_value = ttest_ind(hc_age, mci_age)
    print('Age - HC vs MCI p value {}'.format(p_value))
    _, p_value = ttest_ind(ad_age, mci_age)
    print('Age - MCI vs AD p value {}'.format(p_value))

    contingency_table = pd.crosstab(dataset_corrected_df.Gender, dataset_corrected_df.Diagn)

    _, p_value, _, _ = chi2_contingency(contingency_table[[1, 17]], correction=False)
    print('Gender - HC vs AD p value {}'.format(p_value))
    _, p_value, _, _ = chi2_contingency(contingency_table[[1, 18]], correction=False)
    print('Gender - HC vs AD p value {}'.format(p_value))
    _, p_value, _, _ = chi2_contingency(contingency_table[[17, 18]], correction=False)
    print('Gender - HC vs AD p value {}'.format(p_value))

    hc_age = dataset_corrected_df[dataset_corrected_df['Diagn'] == 1].Age.values
    ad_age = dataset_corrected_df[dataset_corrected_df['Diagn'] == 17].Age.values
    mci_age = dataset_corrected_df[dataset_corrected_df['Diagn'] == 18].Age.values

    contingency_table = pd.crosstab(dataset_corrected_df.Gender, dataset_corrected_df.Diagn)
    print(chi2_contingency(contingency_table, correction=False))
    print(f_oneway(hc_age, ad_age, mci_age))

    homogeneous_df = pd.DataFrame(dataset_corrected_df[dataset_corrected_df['Diagn'].isin([1, 17, 18])])

    hc_age = homogeneous_df[homogeneous_df['Diagn'] == 1].Age.values
    ad_age = homogeneous_df[homogeneous_df['Diagn'] == 17].Age.values
    mci_age = homogeneous_df[homogeneous_df['Diagn'] == 18].Age.values

    contingency_table = pd.crosstab(homogeneous_df.Gender, homogeneous_df.Diagn)
    print(chi2_contingency(contingency_table, correction=False))
    print(f_oneway(hc_age, ad_age, mci_age))

    homogeneous_df = homogeneous_df[['Image_ID']]
    homogeneous_df.to_csv(outputs_dir / (dataset_name + '_homogeneous_ids.csv'), index=False)


if __name__ == "__main__":
    main()
