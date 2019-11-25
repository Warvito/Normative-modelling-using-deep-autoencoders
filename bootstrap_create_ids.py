#!/usr/bin/env python3
"""Script to create the files with the ids of the subjects from UK BIOBANK included in each bootstrap iteration.
These ids are used to train the normative approach.
"""
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path.cwd()


def main():
    """Creates the csv files with the ids of the subjects used to train the normative model."""
    # ----------------------------------------------------------------------------------------
    n_bootstrap = 1000
    ids_path = PROJECT_ROOT / 'outputs' / 'cleaned_ids.csv'
    # ----------------------------------------------------------------------------------------
    # Create experiment's output directory
    outputs_dir = PROJECT_ROOT / 'outputs'
    bootstrap_dir = outputs_dir / 'bootstrap_analysis'
    bootstrap_dir.mkdir(exist_ok=True)

    # Set random seed for random sampling of subjects
    np.random.seed(42)

    ids_df = pd.read_csv(ids_path)
    n_sub = len(ids_df)

    ids_dir = bootstrap_dir / 'ids'
    ids_dir.mkdir(exist_ok=True)

    for i_bootstrap in tqdm(range(n_bootstrap)):
        bootstrap_ids = ids_df.sample(n=n_sub, replace=True)
        ids_filename = 'cleaned_bootstrap_{:03d}.csv'.format(i_bootstrap)
        bootstrap_ids.to_csv(ids_dir / ids_filename, index=False)


if __name__ == "__main__":
    main()
