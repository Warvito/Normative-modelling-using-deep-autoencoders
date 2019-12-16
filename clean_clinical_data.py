#!/usr/bin/env python3
"""Script to clean clinical datasets."""
import argparse
from pathlib import Path

import pandas as pd

from utils import load_demographic_data

PROJECT_ROOT = Path.cwd()


def main(dataset_name):
    """Clean the data from the clinical datasets.

    We removed excluded subjects outside the age range [47,73] based on the UK Biobank data.
    """
    # ----------------------------------------------------------------------------------------
    participants_path = PROJECT_ROOT / 'data' / dataset_name / 'participants.tsv'
    ids_path = PROJECT_ROOT / 'data' / dataset_name / 'freesurferData.csv'

    output_ids_filename = dataset_name + '_cleaned_ids.csv'
    # ----------------------------------------------------------------------------------------
    outputs_dir = PROJECT_ROOT / 'outputs'

    dataset = load_demographic_data(participants_path, ids_path)

    dataset = dataset.loc[(dataset['Age'] >= 47) & (dataset['Age'] <= 73)]

    dataset = dataset.drop_duplicates(subset='participant_id')

    output_ids_df = pd.DataFrame(dataset['participant_id'])
    output_ids_df.to_csv(outputs_dir / output_ids_filename, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--dataset_name',
                        dest='dataset_name',
                        help='Dataset name to clean the data.')
    args = parser.parse_args()

    main(args.dataset_name)
