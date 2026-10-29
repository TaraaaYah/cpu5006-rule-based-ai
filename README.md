# CPU5006 Assessment 1 – Rule-based AI on the Car Evaluation dataset

Compares two rule-based AI algorithms on the UCI Car Evaluation data (1,728 cars, 4 ordinal classes):

| Algorithm | Type | Learns from data? |
|---|---|---|
| **Hierarchical expert system** (`HierarchicalExpertSystem`) | Forward-chaining production system mirroring the documented PRICE / COMFORT / TECH concept hierarchy | No |
| **CN2** (`CN2`) | Multi-class ordered IF-THEN rule-list induction (sequential covering, Laplace heuristic, likelihood-ratio test, ordinal selectors), written from scratch | Yes |

Baselines: majority class and 1R.

## Run
```bash
pip install -r requirements.txt
python run_experiments.py      # about 1 minute; writes results/ and figures/
```

## Files
- `rb_car.py` – algorithms, data loading, metrics (no ML library used for the models)
- `run_experiments.py` – repeated 5-fold CV (x10), significance tests, learning curve, CN2 ablation, figures
- `data/` – `car.data`, `car.names` (UCI Car Evaluation, Bohanec & Rajkovic)
- `results/` – fold-level results, summaries, confusion matrices, rule listings, expert error examples
- `figures/` – figures for the paper

AI-tool use is disclosed in the header of each Python file, as required by the assessment brief.
