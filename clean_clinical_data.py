"""Script to clean clinical datasets.

Based on the BIOBANK's age range, we excluded subjects outside the range [47,73].

"""
from pathlib import Path

import pandas as pd

from utils import load_demographic_data

PROJECT_ROOT = Path.cwd()


def main():
    """"""
    # ----------------------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'
    dataset_name = 'PPMI'

    participants_path = PROJECT_ROOT / 'data' / 'datasets' / dataset_name / 'participants.tsv'
    ids_path = PROJECT_ROOT / 'data' / 'datasets' / dataset_name / 'freesurferData.csv'

    output_ids_filename = dataset_name + '_cleaned_ids.csv'
    # ----------------------------------------------------------------------------------------
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    experiment_dir.mkdir(exist_ok=True)

    dataset = load_demographic_data(participants_path, ids_path)

    dataset = dataset[dataset['Age'] > 46]
    dataset = dataset[dataset['Age'] < 74]

    output_ids_df = pd.DataFrame(dataset['Participant_ID'])
    output_ids_df.to_csv(experiment_dir / output_ids_filename, index=False)


if __name__ == "__main__":
    main()
