"""develop_rules.py - DEVELOPMENT PHASE: look at errors on the development split and refine the rules.

Only the 30% development cars are used here. The 70% test cars are never loaded into any evaluation.

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
import sys
import numpy as np
import pandas as pd
from rb_car import (load_data, evaluate, confusion, CLASSES, COLUMNS, LEVELS,
                    IfThenRuleList, WeightedScoring, MajorityClass)
from split import dev_test_indices

X, y = load_data("data/car.data")
dev, _test_unused = dev_test_indices(y)
Xd, yd = X.iloc[dev].reset_index(drop=True), y[dev]
print(f"Development set: {len(yd)} cars; classes {dict(zip(CLASSES, np.bincount(yd)))}\n")


def show(model, extra=False):
    pred = model.predict(Xd)
    r = evaluate(yd, pred)
    print(f"== {model.name}: accuracy {r['accuracy']:.3f}  macro-F1 {r['macro_f1']:.3f}  kappa {r['kappa']:.3f}  "
          f"ordinal MAE {r['ordinal_mae']:.3f}")
    print("   recall  " + "  ".join(f"{c} {r[f'recall_{c}']:.2f}" for c in CLASSES))
    print("   confusion (rows = true, cols = predicted):")
    print(pd.DataFrame(confusion(yd, pred), index=CLASSES, columns=CLASSES).to_string())
    if extra:
        wrong = pd.DataFrame({c: [LEVELS[c][i] for i in Xd[c]] for c in COLUMNS})
        wrong["true"] = [CLASSES[i] for i in yd]; wrong["pred"] = [CLASSES[i] for i in pred]
        wrong = wrong[wrong.true != wrong.pred]
        print("   most common error patterns (true -> pred):")
        print(wrong.groupby(["true", "pred"]).size().sort_values(ascending=False).head(8).to_string())
    print()
    return pred


if __name__ == "__main__":
    show(MajorityClass().fit(Xd, yd))
    a1 = IfThenRuleList(); p1 = show(a1, extra=True)
    _, which = a1.predict_trace(Xd)
    print("   per-rule behaviour on development cars:")
    for k in range(len(a1.rules) + 1):
        m = which == (k if k < len(a1.rules) else -1)
        if m.sum():
            cls = a1.rules[k][1] if k < len(a1.rules) else a1.default
            print(f"   rule {k + 1}: fires {m.sum():>3}, correct {(yd[m] == cls).mean():.2f}")
    print()
    show(WeightedScoring(), extra=True)
    show(WeightedScoring(use_veto=False, label="  (ablation) scoring without veto"))
    show(WeightedScoring(equal_weights=True, label="  (ablation) equal weights + veto"))
