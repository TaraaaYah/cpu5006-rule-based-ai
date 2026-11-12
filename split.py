"""split.py - the fixed development / test split used throughout (seed 42).

30% of the cars (stratified by class) form the DEVELOPMENT set, used to read errors and refine the
hand-written rules. The other 70% form the TEST set, which is only used once, after the rules are frozen.

AI TOOL DISCLOSURE (required by the assessment brief)
  Tool: Claude (Anthropic, model Sonnet 5.5), used in November 2026.
  What Claude did: wrote the first drafts of the rules, the code in this file and the experiment
  scripts, and refined the rules after reading error tables on the development cars only.
  What I (the author) decided: the dataset, the comparison of two hand-written rule-based
  designs, the 30/70 development/test split, and that the rules were frozen before the single
  test run. My own checks and changes are described in Section 3.7 of the paper.
  Claude had seen results for the whole dataset earlier in the project, so the rules are
  informed by prior knowledge of the data (see the limitations in the paper).
"""
import numpy as np
from sklearn.model_selection import train_test_split  # used only to make the split

SEED = 42


def dev_test_indices(y):
    dev, test = train_test_split(np.arange(len(y)), train_size=0.30, stratify=y, random_state=SEED)
    return np.sort(dev), np.sort(test)
