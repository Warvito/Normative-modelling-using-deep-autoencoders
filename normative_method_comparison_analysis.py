"""Deterministic supervised adversarial autoencoder."""
from pathlib import Path
import random as rn
import time

import numpy as np
from scipy import stats

from utils import COLUMNS_NAME, load_dataset
from models import *

PROJECT_ROOT = Path.cwd()


def ttest_ind_corrected(performance_a, performance_b, k=10, r=10):
    """Corrected repeated k-fold cv test.
     The test assumes that the classifiers were evaluated using cross validation.

    Ref:
        Bouckaert, Remco R., and Eibe Frank. "Evaluating the replicability of significance tests for comparing learning
         algorithms." Pacific-Asia Conference on Knowledge Discovery and Data Mining. Springer, Berlin, Heidelberg, 2004

    Args:
        performance_a: performances from classifier A
        performance_b: performances from classifier B
        k: number of folds
        r: number of repetitions

    Returns:
         t: t-statistic of the corrected test.
         prob: p-value of the corrected test.
    """
    df = k * r - 1

    x = performance_a - performance_b
    m = np.mean(x)

    sigma_2 = np.var(x, ddof=1)
    denom = np.sqrt((1 / k * r + 1 / (k - 1)) * sigma_2)

    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.divide(m, denom)

    prob = stats.t.sf(np.abs(t), df) * 2

    return t, prob


def main():
    """"""
    # ----------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'
    model_names = ['supervised_aae', 'cheung_aae']

    # ----------------------------------------------------------------------------
    # Create directories structure
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    auc_list = []
    for model_name in model_names:
        model_dir = experiment_dir / model_name
        normative_comparison_dir = model_dir / 'normative_comparison'

        auc_scores = np.load(str(normative_comparison_dir / 'auc_scores.npy'))
        auc_list.append(auc_scores)
        print(auc_scores.mean())

    print(ttest_ind_corrected(auc_list[0], auc_list[1], k=10, r=10))


if __name__ == "__main__":
    main()
