"""
run_experiments.py - FINAL EVALUATION on the held-out test cars (70% of the data).

Run this only after the rules in rb_car.py are frozen. It reproduces every number and figure in the paper.
Usage:  python run_experiments.py      (takes a few seconds)
Outputs: results/*.csv, results/*.txt, figures/*.png

AI TOOL DISCLOSURE: drafted with assistance from Claude (Anthropic, Sonnet 5.5), November 2026;
reviewed and run by the student. See the note in rb_car.py.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest

from rb_car import (load_data, evaluate, confusion, CLASSES, COLUMNS, LEVELS, K,
                    IfThenRuleList, WeightedScoring, MajorityClass)
from split import dev_test_indices

N_BOOT, SEED = 2000, 42
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})
COL = {"Majority class": "#999999", "Algorithm 1: IF-THEN rules": "#0072B2", "Algorithm 2: weighted scoring": "#D55E00"}

X, y = load_data("data/car.data")
dev, test = dev_test_indices(y)
Xd, yd = X.iloc[dev].reset_index(drop=True), y[dev]
Xt, yt = X.iloc[test].reset_index(drop=True), y[test]
print(f"Test set: {len(yt)} cars; classes {dict(zip(CLASSES, np.bincount(yt).tolist()))}")

a1, a2 = IfThenRuleList(), WeightedScoring()
N1, N2, N0 = "Algorithm 1: IF-THEN rules", "Algorithm 2: weighted scoring", "Majority class"
models = {N0: MajorityClass().fit(Xd, yd), N1: a1, N2: a2}
# Ablation variants, fixed in advance (not tuned): remove the veto rules; give every attribute equal weight.
ablations = {"Algorithm 2 (full)": a2,
             "Without veto rules": WeightedScoring(use_veto=False),
             "Equal weights, with veto": WeightedScoring(equal_weights=True)}

# ---------------------------------------------------------------------------
# 1. Test-set results with bootstrap 95% confidence intervals
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
boot_idx = rng.integers(0, len(yt), size=(N_BOOT, len(yt)))
preds = {n: m.predict(Xt) for n, m in models.items()}
metrics = ["accuracy", "macro_f1", "kappa", "ordinal_mae"] + [f"recall_{c}" for c in CLASSES] + [f"precision_{c}" for c in CLASSES]
rows, boots = [], {}
for n, p in preds.items():
    base = evaluate(yt, p)
    bs = pd.DataFrame([evaluate(yt[i], p[i]) for i in boot_idx])[metrics]
    boots[n] = bs
    row = {"model": n}
    for m in metrics:
        row[m] = base[m]
        row[m + "_lo"], row[m + "_hi"] = bs[m].quantile(0.025), bs[m].quantile(0.975)
    rows.append(row)
res = pd.DataFrame(rows).set_index("model")
res.round(4).to_csv("results/test_results.csv")
show = res[["accuracy", "macro_f1", "kappa", "ordinal_mae"] + [f"recall_{c}" for c in CLASSES]].round(3)
print(show.to_string())
for n, p in preds.items():
    pd.DataFrame(confusion(yt, p), index=[f"true_{c}" for c in CLASSES], columns=[f"pred_{c}" for c in CLASSES]).to_csv(
        f"results/confusion_test_{n.split(':')[0].replace(' ', '_')}.csv")

# ---------------------------------------------------------------------------
# 2. Comparing the two algorithms on the same cars
# ---------------------------------------------------------------------------
c1, c2 = preds[N1] == yt, preds[N2] == yt
b, c = int((c1 & ~c2).sum()), int((~c1 & c2).sum())  # A1 right/A2 wrong, A1 wrong/A2 right
mcn = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
d_f1 = np.array([evaluate(yt[i], preds[N1][i])["macro_f1"] - evaluate(yt[i], preds[N2][i])["macro_f1"] for i in boot_idx])
d_acc = np.array([(preds[N1][i] == yt[i]).mean() - (preds[N2][i] == yt[i]).mean() for i in boot_idx])
cmp_ = pd.Series({
    "A1_right_A2_wrong": b, "A1_wrong_A2_right": c, "mcnemar_exact_p": mcn,
    "accuracy_diff_A1_minus_A2": res.loc[N1, "accuracy"] - res.loc[N2, "accuracy"],
    "accuracy_diff_lo": np.quantile(d_acc, 0.025), "accuracy_diff_hi": np.quantile(d_acc, 0.975),
    "macro_f1_diff_A1_minus_A2": res.loc[N1, "macro_f1"] - res.loc[N2, "macro_f1"],
    "macro_f1_diff_lo": np.quantile(d_f1, 0.025), "macro_f1_diff_hi": np.quantile(d_f1, 0.975)})
cmp_.round(4).to_csv("results/algorithm_comparison.csv")
print(cmp_.round(4).to_string())

# ---------------------------------------------------------------------------
# 3. Development vs test (generalisation gap), ablations, per-rule behaviour
# ---------------------------------------------------------------------------
gap = []
for n in [N1, N2]:
    d, t = evaluate(yd, models[n].predict(Xd)), evaluate(yt, preds[n])
    for m in ["accuracy", "macro_f1", "kappa"]:
        gap.append({"model": n, "metric": m, "development": d[m], "test": t[m], "drop": d[m] - t[m]})
gap = pd.DataFrame(gap); gap.round(4).to_csv("results/dev_vs_test.csv", index=False); print(gap.round(3).to_string())

abl = pd.DataFrame([{"variant": n, **{k: v for k, v in evaluate(yt, m.predict(Xt)).items()
                                       if k in ("accuracy", "macro_f1", "kappa", "ordinal_mae")}} for n, m in ablations.items()])
abl.round(4).to_csv("results/algorithm2_ablation.csv", index=False); print(abl.round(3).to_string())

_, which = a1.predict_trace(Xt)
per_rule = []
for k in range(len(a1.rules) + 1):
    m = which == (k if k < len(a1.rules) else -1)
    cls = a1.rules[k][1] if k < len(a1.rules) else a1.default
    per_rule.append({"rule": k + 1, "predicts": CLASSES[cls], "fires": int(m.sum()),
                     "correct": int((yt[m] == cls).sum()), "accuracy": float((yt[m] == cls).mean()) if m.sum() else np.nan})
pd.DataFrame(per_rule).round(3).to_csv("results/algorithm1_per_rule_test.csv", index=False)
print(pd.DataFrame(per_rule).round(2).to_string(index=False))


def error_patterns(model, name):
    p = model.predict(Xt)
    df = pd.DataFrame({c: [LEVELS[c][i] for i in Xt[c]] for c in COLUMNS})
    df["true"] = [CLASSES[i] for i in yt]; df["predicted"] = [CLASSES[i] for i in p]
    w = df[df.true != df.predicted]
    s = w.groupby(["true", "predicted"]).size().sort_values(ascending=False).rename("cars").reset_index()
    s.insert(0, "model", name)
    return s
pd.concat([error_patterns(a1, N1), error_patterns(a2, N2)]).to_csv("results/error_patterns_test.csv", index=False)

open("results/algorithm1_rules.txt", "w").write(a1.describe() + "\n")
open("results/algorithm2_rules.txt", "w").write(a2.describe() + "\n")

# ---------------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------------
names = [N0, N1, N2]
short = {N0: "Majority", N1: "Algorithm 1", N2: "Algorithm 2"}
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5), gridspec_kw={"width_ratios": [1.1, 1.5]})
ms = [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("kappa", "Cohen's kappa")]
w = 0.26
for i, n in enumerate(names):
    v = [res.loc[n, m] for m, _ in ms]
    err = [[res.loc[n, m] - res.loc[n, m + "_lo"] for m, _ in ms], [res.loc[n, m + "_hi"] - res.loc[n, m] for m, _ in ms]]
    ax[0].bar(np.arange(3) + (i - 1) * w, v, w, yerr=err, color=COL[n], label=short[n], capsize=2)
ax[0].set_xticks(range(3)); ax[0].set_xticklabels([l for _, l in ms]); ax[0].set_ylim(0, 1.05)
ax[0].legend(frameon=False, fontsize=8, loc="upper right"); ax[0].set_title("(a) Overall performance (95% CI)")
for i, n in enumerate(names):
    ax[1].bar(np.arange(K) + (i - 1) * w, [res.loc[n, f"recall_{c}"] for c in CLASSES], w, color=COL[n])
ax[1].set_xticks(range(K)); ax[1].set_xticklabels(CLASSES); ax[1].set_ylim(0, 1.05)
ax[1].set_ylabel("Recall"); ax[1].set_title("(b) Recall per class")
fig.tight_layout(); fig.savefig("figures/fig1_performance.png"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
for a, n in zip(ax, [N1, N2]):
    m = confusion(yt, preds[n]); rn = m / m.sum(1, keepdims=True)
    a.imshow(rn, cmap="Blues", vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            a.text(j, i, f"{m[i, j]}", ha="center", va="center", color="white" if rn[i, j] > 0.5 else "black", fontsize=9)
    a.set_xticks(range(K)); a.set_xticklabels(CLASSES); a.set_yticks(range(K)); a.set_yticklabels(CLASSES)
    a.set_xlabel("Predicted"); a.set_ylabel("True"); a.set_title(short[n])
    for s in a.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig("figures/fig2_confusion.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.4, 3.3))
for i, (n, lab) in enumerate([(N1, "Algorithm 1"), (N2, "Algorithm 2")]):
    g = gap[(gap.model == n) & (gap.metric == "macro_f1")].iloc[0]
    ax.bar([i - 0.19, i + 0.19], [g.development, g.test], 0.38, color=[COL[n], COL[n]], alpha=1)
    ax.patches[-1].set_alpha(0.55)
    ax.text(i - 0.19, g.development + 0.01, f"{g.development:.2f}", ha="center", fontsize=8)
    ax.text(i + 0.19, g.test + 0.01, f"{g.test:.2f}", ha="center", fontsize=8)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Algorithm 1\n(dark = development, light = test)", "Algorithm 2"], fontsize=8)
ax.set_ylim(0, 1); ax.set_ylabel("Macro-F1"); ax.set_title("Development vs test performance")
fig.tight_layout(); fig.savefig("figures/fig3_dev_vs_test.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.8, 3.3))
x = np.arange(len(abl)); w = 0.36
ax.bar(x - w / 2, abl.accuracy, w, label="Accuracy", color="#56B4E9")
ax.bar(x + w / 2, abl.macro_f1, w, label="Macro-F1", color="#CC79A7")
for i, r in abl.iterrows():
    ax.text(i - w / 2, r.accuracy + 0.01, f"{r.accuracy:.2f}", ha="center", fontsize=8)
    ax.text(i + w / 2, r.macro_f1 + 0.01, f"{r.macro_f1:.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(abl.variant, fontsize=8); ax.set_ylim(0, 1.05)
ax.legend(frameon=False, fontsize=8); ax.set_title("Algorithm 2 ablation (test cars)")
fig.tight_layout(); fig.savefig("figures/fig4_algorithm2_ablation.png"); plt.close(fig)
print("Done.")
