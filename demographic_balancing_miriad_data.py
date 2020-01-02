#!/usr/bin/env python3
"""Script to create homogeneous samples for the MIRIAD dataset.

Labels encoding
"1": "Healthy Controls",
"17": "Alzheimer's Disease",
"""
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind, f_oneway

from utils import load_dataset

PROJECT_ROOT = Path.cwd()


def main():
    """Verify age and gender balance along the groups from the TOMC dataset."""
    # ----------------------------------------------------------------------------------------
    dataset_name = 'MIRIAD'

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

    print(hc_age.mean())
    print(ad_age.mean())

    _, p_value = ttest_ind(hc_age, ad_age)
    print('Age - HC vs AD p value {}'.format(p_value))


    print(chi2_contingency(contingency_table, correction=False))
    print(f_oneway(hc_age, ad_age))

    homogeneous_df = pd.DataFrame(dataset_df[dataset_df['Diagn'].isin([1, 17])].Image_ID)
    homogeneous_df.to_csv(outputs_dir / (dataset_name + '_homogeneous_ids.csv'), index=False)


if __name__ == "__main__":
    main()
