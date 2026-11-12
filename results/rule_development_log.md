# Rule development log (development split only: 518 cars = 30% of the data)

The 1,210 test cars (the other 70%) were not used in any of the numbers below.
Split: `split.py` (stratified by class, seed 42). Command: `python develop_rules.py`.

| Version | What changed | Accuracy | Macro-F1 |
|---|---|---|---|
| Algorithm 1, v1 | First draft: 4 knock-out rules, 1 "vgood" rule, 1 "good" rule, default "acc" | 0.900 | 0.751 |
| Algorithm 1, v2 | After reading dev errors: "good" now needs high safety; added a medium-safety/high-price knock-out; split the "vgood" rule | 0.917 | 0.812 |
| **Algorithm 1, v3 (final)** | Limited the medium-safety knock-out to small/medium boots; added a "vgood" rule for the cheapest purchase price | **0.931** | **0.846** |
| Algorithm 2, v1 | First draft points table and thresholds (85/70/50% of max) | 0.807 | 0.609 |
| **Algorithm 2, v2 (final)** | Price points doubled, safety weight raised, thresholds calibrated on a small grid (54/78/86% of max) | **0.898** | **0.700** |
| Majority class (reference) | Always predicts "unacc" | 0.701 | 0.206 |

Notes
- Algorithm 1 was refined by hand over three rounds by reading error tables on the development cars.
- Algorithm 2's thresholds and point table were chosen by a small grid search over three hand-written
  point tables and a grid of thresholds, scored on the development cars only.
- The AI assistant that drafted the rules had earlier seen results for the full dataset in a previous
  stage of this project, so the rules are informed by prior knowledge of the data. The development/test
  split limits further tuning but does not remove that influence. State this in the limitations.
