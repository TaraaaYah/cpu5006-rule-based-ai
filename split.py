"""split.py - the fixed development / test split used throughout (seed 42).

30% of the cars (stratified by class) form the DEVELOPMENT set, used to read errors and refine the
hand-written rules. The other 70% form the TEST set, which is only used once, after the rules are frozen.
"""
import numpy as np
from sklearn.model_selection import train_test_split  # used only to make the split

SEED = 42


def dev_test_indices(y):
    dev, test = train_test_split(np.arange(len(y)), train_size=0.30, stratify=y, random_state=SEED)
    return np.sort(dev), np.sort(test)
