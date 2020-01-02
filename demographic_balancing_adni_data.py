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

    # ----------------------------------------------------------------------------------------
    outputs_dir = PROJECT_ROOT / 'outputs'
    ids_path = outputs_dir / (dataset_name + '_cleaned_ids.csv')

    dataset_df = load_dataset(participants_path, ids_path, freesurfer_path)
    print(dataset_df.groupby('Diagn').count())

    contingency_table = pd.crosstab(dataset_df.Gender, dataset_df.Diagn)

    _, p_value, _, _ = chi2_contingency(contingency_table[[1, 17]], correction=False)
    print('Gender - HC vs AD p value {}'.format(p_value))
    _, p_value, _, _ = chi2_contingency(contingency_table[[1, 26]], correction=False)
    print('Gender - HC vs SMC p value {}'.format(p_value))
    _, p_value, _, _ = chi2_contingency(contingency_table[[1, 27]], correction=False)
    print('Gender - HC vs EMCI p value {}'.format(p_value))
    _, p_value, _, _ = chi2_contingency(contingency_table[[1, 28]], correction=False)
    print('Gender - HC vs LMCI p value {}'.format(p_value))
    _, p_value, _, _ = chi2_contingency(contingency_table[[17, 26]], correction=False)
    print('Gender - AD vs SMC p value {}'.format(p_value))
    _, p_value, _, _ = chi2_contingency(contingency_table[[17, 27]], correction=False)
    print('Gender - AD vs EMCI p value {}'.format(p_value))
    _, p_value, _, _ = chi2_contingency(contingency_table[[17, 28]], correction=False)
    print('Gender - AD vs LMCI p value {}'.format(p_value))
    _, p_value, _, _ = chi2_contingency(contingency_table[[26, 27]], correction=False)
    print('Gender - SMC vs EMCI p value {}'.format(p_value))
    _, p_value, _, _ = chi2_contingency(contingency_table[[26, 28]], correction=False)
    print('Gender - SMC vs LMCI p value {}'.format(p_value))
    _, p_value, _, _ = chi2_contingency(contingency_table[[27, 28]], correction=False)
    print('Gender - EMCI vs LMCI p value {}'.format(p_value))

    # HC have too many men
    # Removing oldest men to help balancing age
    hc_age = dataset_df[dataset_df['Diagn'] == 1].Age.values

    index_to_remove = dataset_df[(dataset_df['Diagn'] == 1) & (dataset_df['Gender'] == 0)].iloc[hc_age.argmax()].name
    dataset_corrected_df = dataset_df.drop(index_to_remove, axis=0)
    dataset_corrected_df = dataset_corrected_df.reset_index(drop=True)

    for _ in range(55):
        contingency_table = pd.crosstab(dataset_corrected_df.Gender, dataset_corrected_df.Diagn)
        hc_age = dataset_corrected_df[dataset_corrected_df['Diagn'] == 1].Age.values

        _, p_value, _, _ = chi2_contingency(contingency_table[[1, 17]], correction=False)
        print('Gender - HC vs AD p value {}'.format(p_value))
        _, p_value, _, _ = chi2_contingency(contingency_table[[1, 26]], correction=False)
        print('Gender - HC vs SMC p value {}'.format(p_value))
        _, p_value, _, _ = chi2_contingency(contingency_table[[1, 27]], correction=False)
        print('Gender - HC vs EMCI p value {}'.format(p_value))
        _, p_value, _, _ = chi2_contingency(contingency_table[[1, 28]], correction=False)
        print('Gender - HC vs LMCI p value {}'.format(p_value))
        _, p_value, _, _ = chi2_contingency(contingency_table[[17, 26]], correction=False)
        print('Gender - AD vs SMC p value {}'.format(p_value))
        _, p_value, _, _ = chi2_contingency(contingency_table[[17, 27]], correction=False)
        print('Gender - AD vs EMCI p value {}'.format(p_value))
        _, p_value, _, _ = chi2_contingency(contingency_table[[17, 28]], correction=False)
        print('Gender - AD vs LMCI p value {}'.format(p_value))
        _, p_value, _, _ = chi2_contingency(contingency_table[[26, 27]], correction=False)
        print('Gender - SMC vs EMCI p value {}'.format(p_value))
        _, p_value, _, _ = chi2_contingency(contingency_table[[26, 28]], correction=False)
        print('Gender - SMC vs LMCI p value {}'.format(p_value))
        _, p_value, _, _ = chi2_contingency(contingency_table[[27, 28]], correction=False)
        print('Gender - EMCI vs LMCI p value {}'.format(p_value))

        index_to_remove = \
        dataset_corrected_df[(dataset_corrected_df['Diagn'] == 1) & (dataset_corrected_df['Gender'] == 0)].iloc[
            hc_age.argmax()].name
        dataset_corrected_df = dataset_corrected_df.drop(index_to_remove, axis=0)
        dataset_corrected_df = dataset_corrected_df.reset_index(drop=True)

    contingency_table = pd.crosstab(dataset_corrected_df.Gender, dataset_corrected_df.Diagn)

    _, p_value, _, _ = chi2_contingency(contingency_table[[1, 17]], correction=False)
    print('Gender - HC vs AD p value {}'.format(p_value))
    _, p_value, _, _ = chi2_contingency(contingency_table[[1, 26]], correction=False)
    print('Gender - HC vs SMC p value {}'.format(p_value))
    _, p_value, _, _ = chi2_contingency(contingency_table[[1, 27]], correction=False)
    print('Gender - HC vs EMCI p value {}'.format(p_value))
    _, p_value, _, _ = chi2_contingency(contingency_table[[1, 28]], correction=False)
    print('Gender - HC vs LMCI p value {}'.format(p_value))
    _, p_value, _, _ = chi2_contingency(contingency_table[[17, 26]], correction=False)
    print('Gender - AD vs SMC p value {}'.format(p_value))
    _, p_value, _, _ = chi2_contingency(contingency_table[[17, 27]], correction=False)
    print('Gender - AD vs EMCI p value {}'.format(p_value))
    _, p_value, _, _ = chi2_contingency(contingency_table[[17, 28]], correction=False)
    print('Gender - AD vs LMCI p value {}'.format(p_value))
    _, p_value, _, _ = chi2_contingency(contingency_table[[26, 27]], correction=False)
    print('Gender - SMC vs EMCI p value {}'.format(p_value))
    _, p_value, _, _ = chi2_contingency(contingency_table[[26, 28]], correction=False)
    print('Gender - SMC vs LMCI p value {}'.format(p_value))
    _, p_value, _, _ = chi2_contingency(contingency_table[[27, 28]], correction=False)
    print('Gender - EMCI vs LMCI p value {}'.format(p_value))

    hc_age = dataset_corrected_df[dataset_corrected_df['Diagn'] == 1].Age.values
    ad_age = dataset_corrected_df[dataset_corrected_df['Diagn'] == 17].Age.values
    smc_age = dataset_corrected_df[dataset_corrected_df['Diagn'] == 26].Age.values
    emci_age = dataset_corrected_df[dataset_corrected_df['Diagn'] == 27].Age.values
    lmci_age = dataset_corrected_df[dataset_corrected_df['Diagn'] == 28].Age.values

    _, p_value = ttest_ind(hc_age, ad_age)
    print('Age - HC vs AD p value {}'.format(p_value))
    _, p_value = ttest_ind(hc_age, smc_age)
    print('Age - HC vs SMC p value {}'.format(p_value))
    _, p_value = ttest_ind(hc_age, emci_age)
    print('Age - HC vs EMCI p value {}'.format(p_value))
    _, p_value = ttest_ind(hc_age, lmci_age)
    print('Age - HC vs LMCI p value {}'.format(p_value))
    _, p_value = ttest_ind(ad_age, smc_age)
    print('Age - AD vs SMC p value {}'.format(p_value))
    _, p_value = ttest_ind(ad_age, emci_age)
    print('Age - AD vs EMCI p value {}'.format(p_value))
    _, p_value = ttest_ind(ad_age, lmci_age)
    print('Age - AD vs LMCI p value {}'.format(p_value))
    _, p_value = ttest_ind(smc_age, emci_age)
    print('Age - SMC vs EMCI p value {}'.format(p_value))
    _, p_value = ttest_ind(smc_age, lmci_age)
    print('Age - SMC vs LMCI p value {}'.format(p_value))
    _, p_value = ttest_ind(emci_age, lmci_age)
    print('Age - EMCI vs LMCI p value {}'.format(p_value))

    print(hc_age.mean())
    print(ad_age.mean())
    print(smc_age.mean())
    print(emci_age.mean())
    print(lmci_age.mean())

    # hc is too old, dropping some of the oldest
    print(hc_age.argmax())
    print(dataset_corrected_df[dataset_corrected_df['Diagn'] == 1].iloc[hc_age.argmax()].Age)
    print(dataset_corrected_df[dataset_corrected_df['Diagn'] == 1].iloc[hc_age.argmax()].name)
    print('')

    dataset_corrected_df = dataset_corrected_df.drop(
        dataset_corrected_df[dataset_corrected_df['Diagn'] == 1].iloc[hc_age.argmax()].name, axis=0)

    for _ in range(21):
        hc_age = dataset_corrected_df[dataset_corrected_df['Diagn'] == 1].Age.values

        print(hc_age.argmax())
        print(dataset_corrected_df[dataset_corrected_df['Diagn'] == 1].iloc[hc_age.argmax()].Age)
        print(dataset_corrected_df[dataset_corrected_df['Diagn'] == 1].iloc[hc_age.argmax()].Image_ID)
        print(hc_age.argmax())
        print('')

        _, p_value = ttest_ind(hc_age, ad_age)
        print('Age - HC vs AD p value {}'.format(p_value))
        _, p_value = ttest_ind(hc_age, smc_age)
        print('Age - HC vs SMC p value {}'.format(p_value))
        _, p_value = ttest_ind(hc_age, emci_age)
        print('Age - HC vs EMCI p value {}'.format(p_value))
        _, p_value = ttest_ind(hc_age, lmci_age)
        print('Age - HC vs LMCI p value {}'.format(p_value))

        print(hc_age.mean())
        print(ad_age.mean())
        print(smc_age.mean())
        print(emci_age.mean())
        print(lmci_age.mean())

        dataset_corrected_df = dataset_corrected_df.drop(
            dataset_corrected_df[dataset_corrected_df['Diagn'] == 1].iloc[hc_age.argmax()].name, axis=0)
        dataset_corrected_df = dataset_corrected_df.reset_index(drop=True)

    hc_age = dataset_corrected_df[dataset_corrected_df['Diagn'] == 1].Age.values
    ad_age = dataset_corrected_df[dataset_corrected_df['Diagn'] == 17].Age.values
    emci_age = dataset_corrected_df[dataset_corrected_df['Diagn'] == 27].Age.values
    lmci_age = dataset_corrected_df[dataset_corrected_df['Diagn'] == 28].Age.values

    contingency_table = pd.crosstab(dataset_corrected_df.Gender, dataset_corrected_df.Diagn)
    print(chi2_contingency(contingency_table, correction=False))
    print(f_oneway(hc_age, ad_age, emci_age, lmci_age))

    homogeneous_df = pd.DataFrame(dataset_corrected_df[dataset_corrected_df['Diagn'].isin([1, 17, 27, 28])])

    hc_age = homogeneous_df[homogeneous_df['Diagn'] == 1].Age.values
    ad_age = homogeneous_df[homogeneous_df['Diagn'] == 17].Age.values
    emci_age = homogeneous_df[homogeneous_df['Diagn'] == 27].Age.values
    lmci_age = homogeneous_df[homogeneous_df['Diagn'] == 28].Age.values

    contingency_table = pd.crosstab(homogeneous_df.Gender, homogeneous_df.Diagn)
    print(chi2_contingency(contingency_table, correction=False))
    print(f_oneway(hc_age, ad_age, emci_age, lmci_age))

    homogeneous_df = homogeneous_df[['Image_ID']]
    homogeneous_df.to_csv(outputs_dir / (dataset_name + '_homogeneous_ids.csv'), index=False)


if __name__ == "__main__":
    main()
