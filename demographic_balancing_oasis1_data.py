#!/usr/bin/env python3
"""Script to create an homogeneous sample for the OASIS dataset.

Labels encoding
"1": "Healthy Controls",
"17": "Alzheimer's Disease",
"""
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind, f_oneway

from utils import load_dataset

PROJECT_ROOT = Path.cwd()


def main():
    """Verify age and gender balance along the groups from the ADNI dataset."""
    # ----------------------------------------------------------------------------------------
    dataset_name = 'OASIS1'

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

    hc_age = dataset_df[dataset_df['Diagn'] == 1].Age.values
    ad_age = dataset_df[dataset_df['Diagn'] == 17].Age.values

    _, p_value = ttest_ind(hc_age, ad_age)
    print('Age - HC vs AD p value {}'.format(p_value))

    print(hc_age.mean())
    print(ad_age.mean())

    # HC is too young, dropping some of the youngest
    print(hc_age.argmin())
    print(dataset_df[dataset_df['Diagn'] == 1].iloc[hc_age.argmin()].Age)
    print(dataset_df[dataset_df['Diagn'] == 1].iloc[hc_age.argmin()].name)
    print('')

    dataset_corrected_df = dataset_df.drop(dataset_df[dataset_df['Diagn'] == 1].iloc[hc_age.argmin()].name, axis=0)
    hc_age = np.delete(hc_age, hc_age.argmin(), 0)

    for _ in range(48):
        print(hc_age.argmin())
        print(dataset_corrected_df[dataset_corrected_df['Diagn'] == 1].iloc[hc_age.argmin()].Age)
        print(dataset_corrected_df[dataset_corrected_df['Diagn'] == 1].iloc[hc_age.argmin()].name)
        print(hc_age)
        print('')
        dataset_corrected_df = dataset_corrected_df.drop(
            dataset_corrected_df[dataset_corrected_df['Diagn'] == 1].iloc[hc_age.argmin()].name, axis=0)
        hc_age = np.delete(hc_age, hc_age.argmin(), 0)
        dataset_corrected_df = dataset_corrected_df.reset_index(drop=True)

    _, p_value = ttest_ind(hc_age, ad_age)
    print('Age - HC vs AD p value {}'.format(p_value))

    hc_age = dataset_corrected_df[dataset_corrected_df['Diagn'] == 1].Age.values
    ad_age = dataset_corrected_df[dataset_corrected_df['Diagn'] == 17].Age.values

    contingency_table = pd.crosstab(dataset_corrected_df.Gender, dataset_corrected_df.Diagn)
    print(chi2_contingency(contingency_table, correction=False))
    print(f_oneway(hc_age, ad_age))

    homogeneous_df = pd.DataFrame(dataset_corrected_df[dataset_corrected_df['Diagn'].isin([1, 17])].Image_ID)
    homogeneous_df.to_csv(outputs_dir / (dataset_name + '_homogeneous_ids.csv'), index=False)


if __name__ == "__main__":
    main()
