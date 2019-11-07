"""Clean UK Biobank scanner 1 (Cheadle) data.

Subjects from the Assessment Centre from Cheadle (code 11025) are majority white.
Besides, some ages have very low number of subjects (<100). The ethnics minorities
and age with low number are remove from further analysis as well subjects with any
mental or brain disorder.

"""
from pathlib import Path

import pandas as pd

from utils import load_demographic_data

PROJECT_ROOT = Path.cwd()


def main():
    """Clean UK Biobank scanner 1 data."""
    # ----------------------------------------------------------------------------------------
    demographic_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'ukb22321.csv'
    participants_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'participants.tsv'
    ids_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'freesurferData.csv'

    output_ids_filename = 'cleaned_ids.csv'
    # ----------------------------------------------------------------------------------------
    # Create experiment's output directory
    outputs_dir = PROJECT_ROOT / 'outputs'
    outputs_dir.mkdir(exist_ok=True)

    dataset = load_demographic_data(demographic_path, ids_path)

    participants_df = pd.read_csv(participants_path, sep='\t', usecols=['Participant_ID', 'Diagn'])
    participants_df['ID'] = participants_df['Participant_ID'].str.split('-').str[1]
    participants_df['ID'] = pd.to_numeric(participants_df['ID'])

    dataset = pd.merge(dataset, participants_df, on='ID')

    # Exclude ages with <100 participants,
    dataset = dataset.loc[dataset['Age'] > 46]

    # Exclude non-white ethnicities due to small subgroups
    dataset = dataset.loc[dataset['Ethnicity'] == 'White']

    # Exclude patients
    dataset = dataset.loc[dataset['Diagn'] == 1]

    output_ids_df = pd.DataFrame(dataset['Participant_ID'])
    output_ids_df.to_csv(outputs_dir / output_ids_filename, index=False)


if __name__ == "__main__":
    main()
