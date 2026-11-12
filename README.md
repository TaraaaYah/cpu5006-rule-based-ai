# CPU5006 Assessment 1 – Rule-based AI on the Car Evaluation dataset

Compares two hand-written rule-based AI systems on the UCI Car Evaluation data (1,728 cars, 4 ordinal classes):

| Algorithm | Description |
|---|---|
| **Algorithm 1** (`IfThenRuleList`) | A flat, ordered list of IF-THEN rules; the first matching rule decides. |
| **Algorithm 2** (`WeightedScoring`) | A points-based scoring system with explicit veto rules and class thresholds. |

Baseline: always predict the majority class. Neither algorithm learns from data; every rule and point value is written by hand.

## Method
- Cars are split 30% development / 70% test (`split.py`, stratified, seed 42).
- The rules were refined on the development cars only (`develop_rules.py`, `results/rule_development_log.md`), then frozen.
- The 1,210 test cars were used once (`run_experiments.py`): accuracy, macro-F1, kappa, per-class recall, bootstrap 95% CIs, an exact McNemar test and an ablation of Algorithm 2.

## Headline test results
| System | Accuracy | Macro-F1 |
|---|---|---|
| Always "unacc" | 0.700 | 0.206 |
| Algorithm 1: IF-THEN rules | 0.909 | 0.796 |
| Algorithm 2: weighted scoring | 0.895 | 0.689 |

## Files
- `rb_car.py` – both algorithms, the baseline, data loading and metrics
- `split.py` – the fixed 30% development / 70% test split (seed 42)
- `develop_rules.py` – development-phase evaluation on the development cars only
- `run_experiments.py` – final test evaluation, statistics and figures
- `data/` – `car.data`, `car.names` (UCI Car Evaluation, Bohanec and Rajkovic)
- `results/` – test results, comparisons, rule listings, error analysis and the rule development log
- `figures/` – the four figures used in the paper
- `PAPER_GUIDE.md` – notes for writing the paper

## Run
```bash
pip install -r requirements.txt
python develop_rules.py     # development phase (development cars only)
python run_experiments.py   # final test evaluation, results and figures
```

AI-tool use is disclosed in the header of each Python file, as required by the assessment brief.
