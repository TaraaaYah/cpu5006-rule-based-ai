"""
run_experiments.py - reproduces every number and figure for the paper (Car Evaluation).

Usage:  python run_experiments.py      (a few minutes)
Outputs: results/*.csv, results/*.txt, figures/*.png

AI TOOL DISCLOSURE: drafted with assistance from Claude (Anthropic, Sonnet 5.5),
October 2026; reviewed and run by the student. See the note in rb_car.py.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split  # splits only

from rb_car import (load_data, evaluate, confusion, MajorityClass, OneR,
                    HierarchicalExpertSystem, CN2, CLASSES, K)

SEED, N_SPLITS, N_REPEATS = 42, 5, 10
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})
COL = {"Majority class": "#999999", "1R": "#E69F00", "Expert system": "#0072B2", "CN2": "#D55E00"}
FACT = {"Majority class": MajorityClass, "1R": OneR, "Expert system": HierarchicalExpertSystem, "CN2": CN2}


def cross_validate(X, y, factories, n_repeats=N_REPEATS, keep_cm=False):
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=n_repeats, random_state=SEED)
    rows, cms = [], {k: np.zeros((K, K), int) for k in factories}
    for i, (tr, te) in enumerate(rskf.split(X, y)):
        rep, fold = divmod(i, N_SPLITS)
        for name, make in factories.items():
            m = make().fit(X.iloc[tr], y[tr])
            pred = m.predict(X.iloc[te])
            r = evaluate(y[te], pred)
            r.update(model=name, repeat=rep, fold=fold, n_rules=m.n_rules(), avg_rule_len=m.avg_len())
            rows.append(r)
            if keep_cm and rep == 0:
                cms[name] += confusion(y[te], pred)
    return pd.DataFrame(rows), cms


def nadeau_bengio(d, test_frac):
    """Corrected resampled t-test (Nadeau & Bengio, 2003) on paired per-fold differences."""
    d = np.asarray(d, float); k = len(d)
    if d.var(ddof=1) == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / k + test_frac / (1 - test_frac)) * d.var(ddof=1))
    return t, 2 * stats.t.sf(abs(t), k - 1)


# ---------------------------------------------------------------------------
# 1. Main experiment
# ---------------------------------------------------------------------------
X, y = load_data("data/car.data")
print(f"{len(y)} cars; class counts {dict(zip(CLASSES, np.bincount(y)))}")
folds, cms = cross_validate(X, y, FACT, keep_cm=True)
folds.to_csv("results/cv_fold_results.csv", index=False)
metrics = ["accuracy", "macro_f1", "kappa", "ordinal_mae"] + [f"recall_{c}" for c in CLASSES] + ["n_rules", "avg_rule_len"]
per_rep = folds.groupby(["model", "repeat"])[metrics].mean().reset_index()
summ = per_rep.groupby("model")[metrics].agg(["mean", "std"])
summ.columns = [f"{a}_{b}" for a, b in summ.columns]
summ = summ.reindex(list(FACT))
summ.round(4).to_csv("results/summary.csv")
print(summ[[f"{m}_mean" for m in metrics]].round(3).to_string())
for n, m in cms.items():
    pd.DataFrame(m, index=[f"true_{c}" for c in CLASSES], columns=[f"pred_{c}" for c in CLASSES]).to_csv(
        f"results/confusion_{n.replace(' ', '_')}.csv")

# Expert system needs no training: also score it on the whole dataset once.
es = HierarchicalExpertSystem()
full = evaluate(y, es.predict(X))
pd.Series(full).round(4).to_csv("results/expert_full_dataset.csv")
print("Expert system on all 1,728 cars:", {k: round(v, 3) for k, v in full.items() if k in ("accuracy", "macro_f1", "kappa")})

# ---------------------------------------------------------------------------
# 2. Significance (Nadeau-Bengio), paired per-fold
# ---------------------------------------------------------------------------
sig_rows = []
for a, b in [("CN2", "Expert system"), ("CN2", "1R"), ("Expert system", "1R")]:
    A = folds[folds.model == a].sort_values(["repeat", "fold"]).reset_index(drop=True)
    B = folds[folds.model == b].sort_values(["repeat", "fold"]).reset_index(drop=True)
    for m in ["accuracy", "macro_f1", "kappa", "ordinal_mae"]:
        t, p = nadeau_bengio(A[m] - B[m], 1 / N_SPLITS)
        sig_rows.append({"A": a, "B": b, "metric": m, "mean_diff_A_minus_B": (A[m] - B[m]).mean(), "t": t, "p": p})
sig = pd.DataFrame(sig_rows); sig.to_csv("results/significance.csv", index=False)
print(sig.round(4).to_string())

# ---------------------------------------------------------------------------
# 3. Learning curve: how many labelled cars does each method need?
# ---------------------------------------------------------------------------
FRACS = [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
lc = []
for seed in range(10):
    tr_all, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    rng = np.random.default_rng(seed)
    for f in FRACS:
        tr = rng.choice(tr_all, size=max(int(f * len(tr_all)), 5), replace=False)
        for name in ["1R", "CN2", "Expert system"]:
            m = FACT[name]().fit(X.iloc[tr], y[tr])
            r = evaluate(y[te], m.predict(X.iloc[te]))
            lc.append({"model": name, "frac": f, "n_train": len(tr), "seed": seed,
                       "accuracy": r["accuracy"], "macro_f1": r["macro_f1"], "n_rules": m.n_rules()})
lc = pd.DataFrame(lc); lc.groupby(["model", "n_train"]).mean(numeric_only=True).drop(columns=["frac", "seed"]).round(4).to_csv(
    "results/learning_curve.csv")
print(lc.groupby(["model", "frac"])[["accuracy", "macro_f1", "n_rules"]].mean().round(3).unstack(0)["macro_f1"])

# ---------------------------------------------------------------------------
# 4. CN2 sensitivity / ablation: ordinal selectors, max length, min coverage
# ---------------------------------------------------------------------------
sens = []
for ordinal in [True, False]:
    for ml in [2, 3, 4]:
        for mc in [2, 5, 10, 20]:
            f, _ = cross_validate(X, y, {"CN2": lambda o=ordinal, l=ml, c=mc: CN2(max_len=l, min_cov=c, ordinal=o)}, n_repeats=2)
            s = f[["accuracy", "macro_f1", "kappa", "n_rules"]].mean()
            sens.append({"ordinal": ordinal, "max_len": ml, "min_cov": mc, **s.to_dict()})
            print("sens", ordinal, ml, mc, s.round(3).to_dict())
sens = pd.DataFrame(sens); sens.to_csv("results/cn2_sensitivity.csv", index=False)

# ---------------------------------------------------------------------------
# 5. Final rule listings (CN2 fitted to all 1,728 cars; expert rules as written)
# ---------------------------------------------------------------------------
cn2 = CN2().fit(X, y)
open("results/induced_rules_cn2.txt", "w").write(cn2.describe() + "\n")
open("results/expert_rules.txt", "w").write(es.describe() + "\n")
errs = [(i, es.explain(X.iloc[i])) for i in np.where(es.predict(X) != y)[0][:15]]
pd.DataFrame({"row": [i for i, _ in errs], "true": [CLASSES[y[i]] for i, _ in errs],
              "reasoning": [t for _, t in errs]}).to_csv("results/expert_error_examples.csv", index=False)

# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------
names = list(summ.index)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1.2, 1.6]})
ms = [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("kappa", "Cohen's kappa")]
w = 0.2
for i, n in enumerate(names):
    ax[0].bar(np.arange(3) + (i - 1.5) * w, [summ.loc[n, f"{m}_mean"] for m, _ in ms], w,
              yerr=[summ.loc[n, f"{m}_std"] for m, _ in ms], color=COL[n], label=n, capsize=2)
ax[0].set_xticks(range(3)); ax[0].set_xticklabels([l for _, l in ms]); ax[0].set_ylim(0, 1.05)
ax[0].legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2); ax[0].set_title("(a) Overall performance")
ax[0].set_ylabel("Mean of 10 repeats of 5-fold CV")
for i, n in enumerate(names):
    ax[1].bar(np.arange(K) + (i - 1.5) * w, [summ.loc[n, f"recall_{c}_mean"] for c in CLASSES], w, color=COL[n])
ax[1].set_xticks(range(K)); ax[1].set_xticklabels(CLASSES); ax[1].set_ylim(0, 1.05)
ax[1].set_ylabel("Recall"); ax[1].set_title("(b) Recall per class (minority classes are hard)")
fig.tight_layout(); fig.savefig("figures/fig1_performance.png"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
for a, n in zip(ax, ["Expert system", "CN2"]):
    m = cms[n]; rn = m / m.sum(1, keepdims=True)
    a.imshow(rn, cmap="Blues", vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            a.text(j, i, f"{m[i, j]}", ha="center", va="center", color="white" if rn[i, j] > 0.5 else "black", fontsize=9)
    a.set_xticks(range(K)); a.set_xticklabels(CLASSES); a.set_yticks(range(K)); a.set_yticklabels(CLASSES)
    a.set_xlabel("Predicted"); a.set_ylabel("True"); a.set_title(n)
    for s in a.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig("figures/fig2_confusion.png"); plt.close(fig)

g = lc.groupby(["model", "n_train"])[["accuracy", "macro_f1", "n_rules"]].mean().reset_index()
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.3))
for n in ["1R", "CN2", "Expert system"]:
    d = g[g.model == n]
    ax[0].plot(d.n_train, d.macro_f1, "o-", color=COL[n], label=n)
    ax[1].plot(d.n_train, d.n_rules, "o-", color=COL[n], label=n)
ax[0].set_xscale("log"); ax[1].set_xscale("log")
ax[0].set_xlabel("Labelled training cars (log scale)"); ax[0].set_ylabel("Macro-F1 on held-out 20%")
ax[1].set_xlabel("Labelled training cars (log scale)"); ax[1].set_ylabel("Rules in model")
ax[0].legend(frameon=False, fontsize=8); ax[0].set_title("(a) Learning curve"); ax[1].set_title("(b) Model size")
fig.tight_layout(); fig.savefig("figures/fig3_learning_curve.png"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(8, 3.3), sharey=True)
for a, ml in zip(ax, [3, 4]):
    for o, mk in [(True, "o-"), (False, "s--")]:
        d = sens[(sens.ordinal == o) & (sens.max_len == ml)].sort_values("min_cov")
        a.plot(d.min_cov, d.macro_f1, mk, label="ordinal selectors" if o else "equality selectors only")
    a.set_xticks([2, 5, 10, 20])
    a.set_xlabel("Minimum coverage"); a.set_title(f"Maximum rule length {ml}")
ax[0].set_ylabel("Macro-F1"); ax[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("figures/fig4_cn2_ablation.png"); plt.close(fig)
print("Done.")
