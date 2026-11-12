# CPU5006 Assessment 1 – Rule-based AI on the Car Evaluation dataset

Compares two hand-written rule-based AI systems on the UCI Car Evaluation data (1,728 cars, 4 ordinal classes):

| Algorithm | Description |
|---|---|
| **Algorithm 1** (`IfThenRuleList`) | A flat, ordered list of IF-THEN rules; the first matching rule decides. |
| **Algorithm 2** (`WeightedScoring`) | A points-based scoring system with explicit veto rules and class thresholds. |

Baseline: always predict the majority class. Neither algorithm learns from data; every rule and point value is written by hand.

## Status: development phase
- The rules were refined on a 30% development split (`split.py`, `develop_rules.py`); see `results/rule_development_log.md`.
- The 70% test split has not been used yet. The final evaluation script and results will be added once the rules are frozen.

## Files
- `rb_car.py` – both algorithms, the baseline, data loading and metrics
- `split.py` – the fixed 30% development / 70% test split (seed 42)
- `develop_rules.py` – development-phase evaluation on the development cars only
- `data/` – `car.data`, `car.names` (UCI Car Evaluation, Bohanec and Rajkovic)
- `results/rule_development_log.md` – version history of the rules

## Run
```bash
pip install -r requirements.txt
python develop_rules.py
```

AI-tool use is disclosed in the header of each Python file, as required by the assessment brief.
