#!/usr/bin/env python3
"""Clean UK Biobank scanner 1 (Cheadle) data.

Subjects from the Assessment Centre from Cheadle (code 11025) are majority white.
Besides, some ages have very low number of subjects (<100). The ethnics minorities
and age with low number are remove from further analysis as well subjects with any
mental or brain disorder.

"""
from pathlib import Path

from utils import load_demographic_data

PROJECT_ROOT = Path.cwd()


def main():
    """Clean UK Biobank scanner 1 data."""
    # ----------------------------------------------------------------------------------------
    participants_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'participants.tsv'
    ids_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'freesurferData.csv'

    output_ids_filename = 'cleaned_ids.csv'
    # ----------------------------------------------------------------------------------------
    # Create experiment's output directory
    outputs_dir = PROJECT_ROOT / 'outputs'
    outputs_dir.mkdir(exist_ok=True)

    dataset = load_demographic_data(participants_path, ids_path)

    # Exclude subjects outside [47, 73] interval (ages with <100 participants).
    dataset = dataset.loc[(dataset['Age'] >= 47) & (dataset['Age'] <= 73)]

    # Exclude non-white ethnicities due to small subgroups
    dataset = dataset.loc[dataset['Ethnicity'] == 'White']

    # Exclude scanner02
    dataset = dataset.loc[dataset['Dataset'] == 'BIOBANK-SCANNER01']

    # Exclude subjects with previous hospitalization
    dataset = dataset.loc[dataset['Diagn'] == 1]

    output_ids_df = dataset[['Image_ID']]

    assert sum(output_ids_df.duplicated()) == 0

    output_ids_df.to_csv(outputs_dir / output_ids_filename, index=False)


if __name__ == "__main__":
    main()
