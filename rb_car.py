"""
rb_car.py - Two hand-written rule-based AI systems for the Car Evaluation dataset
=================================================================================
CPU5006 Artificial Intelligence, Assessment 1 (Rule-Based AI Experiments).

  1. IfThenRuleList  - a flat, ordered list of hand-written IF-THEN rules
                       (first matching rule fires; conflict resolution by order).
  2. WeightedScoring - a points-based scoring system: every attribute value earns
                       points, the total is compared with class thresholds, and
                       explicit veto rules sit on top of the score.
  3. MajorityClass   - trivial baseline.

Neither algorithm learns from data: every rule, point value and threshold is
written by hand and can be read and changed in this file.

AI TOOL DISCLOSURE (required by the assessment brief)
  Tool: Claude (Anthropic, model Sonnet 5.5), used in November 2026.
  What Claude did: wrote the first drafts of the rules, the code in this file and the experiment
  scripts, and refined the rules after reading error tables on the development cars only.
  What I (the author) decided: the dataset, the comparison of two hand-written rule-based
  designs, the 30/70 development/test split, and that the rules were frozen before the single
  test run. My own checks and changes are described in Section 3.7 of the paper.
  Claude had seen results for the whole dataset earlier in the project, so the rules are
  informed by prior knowledge of the data (see the limitations in the paper).

Only numpy / pandas are used. No machine-learning library is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
# Ordinal level order for every attribute (rank 0 = least desirable for the buyer).
LEVELS = {
    "buying": ["vhigh", "high", "med", "low"],       # a low price is best
    "maint": ["vhigh", "high", "med", "low"],
    "doors": ["2", "3", "4", "5more"],
    "persons": ["2", "4", "more"],
    "lug_boot": ["small", "med", "big"],
    "safety": ["low", "med", "high"],
}
CLASSES = ["unacc", "acc", "good", "vgood"]  # ordinal: unacceptable < acceptable < good < very good
K = len(CLASSES)


def load_data(path):
    """Return (X, y): X holds ordinal ranks (ints) per attribute, y holds class ranks 0..3."""
    df = pd.read_csv(path, header=None, names=COLUMNS + ["class"], dtype=str)
    df = df.apply(lambda s: s.str.strip().str.replace("-", ""))
    X = pd.DataFrame({c: df[c].map({v: i for i, v in enumerate(LEVELS[c])}) for c in COLUMNS})
    y = df["class"].map({v: i for i, v in enumerate(CLASSES)}).to_numpy()
    assert not X.isna().any().any() and not np.isnan(y.astype(float)).any()
    return X.astype(int), y.astype(int)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def confusion(y, pred):
    m = np.zeros((K, K), int)
    np.add.at(m, (np.asarray(y), np.asarray(pred)), 1)
    return m


def evaluate(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    m = confusion(y, pred)
    n = m.sum()
    tp = np.diag(m).astype(float)
    rec = np.divide(tp, m.sum(1), out=np.zeros(K), where=m.sum(1) > 0)
    prec = np.divide(tp, m.sum(0), out=np.zeros(K), where=m.sum(0) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(K), where=(prec + rec) > 0)
    po = tp.sum() / n
    pe = (m.sum(0) * m.sum(1)).sum() / n ** 2
    out = {"accuracy": po, "macro_f1": f1.mean(), "kappa": (po - pe) / (1 - pe) if pe < 1 else 0.0,
           "ordinal_mae": float(np.abs(y - pred).mean())}
    for k, c in enumerate(CLASSES):
        out[f"recall_{c}"], out[f"precision_{c}"] = rec[k], prec[k]
    return out


# --------------------------------------------------------------------------
# A tiny rule language shared by both systems.
# A condition is (attribute, operator, level) with operator '==', '<=' or '>='
# compared on the attribute's ordinal rank, e.g. ("safety", ">=", "med").
# --------------------------------------------------------------------------
def _rank(attr, level):
    return LEVELS[attr].index(level)


def cond_mask(X, cond):
    attr, op, level = cond
    col, r = X[attr].to_numpy(), _rank(attr, level)
    return col == r if op == "==" else (col <= r if op == "<=" else col >= r)


def cond_text(cond):
    attr, op, level = cond
    words = {"==": "is", "<=": "is at most", ">=": "is at least"}
    return f"{attr} {words[op]} {level}"


def rule_mask(X, conds):
    m = np.ones(len(X), dtype=bool)
    for c in conds:
        m &= cond_mask(X, c)
    return m


# --------------------------------------------------------------------------
# Baseline
# --------------------------------------------------------------------------
class MajorityClass:
    name = "Majority class"

    def fit(self, X, y):
        self.cls = int(np.bincount(y, minlength=K).argmax())
        return self

    def predict(self, X):
        return np.full(len(X), self.cls)


# --------------------------------------------------------------------------
# Algorithm 1: flat, ordered IF-THEN rule list (hand-written)
# --------------------------------------------------------------------------
# Each rule: (conditions, class, reason). Rules are tried in order; the first
# rule whose conditions all hold decides the class. RULES_VERSION records how
# many refinement rounds were made on the development split (see results log).
IF_THEN_RULES = [
    # Knock-out rules: a car failing one of these is unacceptable whatever else is good.
    ([("persons", "==", "2")], 0, "A two-seater cannot carry a family"),
    ([("safety", "==", "low")], 0, "Low safety is never acceptable"),
    ([("buying", "==", "vhigh"), ("maint", "<=", "high")], 0, "Very high price plus high running costs"),
    ([("maint", "==", "vhigh"), ("buying", "<=", "high")], 0, "Very high running costs plus a high price"),
    ([("safety", "==", "med"), ("buying", "==", "high"), ("lug_boot", "<=", "med")], 0,
     "Medium safety, a high price and limited boot space"),
    # Very good cars: top safety, cheap to buy, roomy boot.
    ([("safety", "==", "high"), ("lug_boot", ">=", "med"), ("buying", ">=", "med"), ("maint", ">=", "med")], 3,
     "Safest, roomy boot, cheap to buy and run"),
    ([("safety", "==", "high"), ("buying", "==", "low"), ("maint", "==", "high"), ("lug_boot", ">=", "med")], 3,
     "Safest, roomy boot, cheapest to buy"),
    # Good cars: top safety and cheap to buy and run, or medium safety but the cheapest to buy and run.
    ([("safety", "==", "high"), ("buying", ">=", "med"), ("maint", ">=", "med")], 2,
     "Safest and cheap to buy and run"),
    ([("safety", "==", "med"), ("buying", "==", "low"), ("maint", "==", "low")], 2,
     "Medium safety but the cheapest to buy and run"),
    # Everything else that survived the knock-outs is acceptable (the default below).
]
IF_THEN_DEFAULT = 1  # acc


class IfThenRuleList:
    name = "Algorithm 1: IF-THEN rule list"

    def __init__(self, rules=None, default=None):
        self.rules = IF_THEN_RULES if rules is None else rules
        self.default = IF_THEN_DEFAULT if default is None else default

    def fit(self, X, y):  # nothing to learn; kept for a common interface
        return self

    def predict_trace(self, X):
        n = len(X)
        pred = np.full(n, self.default)
        which = np.full(n, -1)
        done = np.zeros(n, dtype=bool)
        for k, (conds, cls, _why) in enumerate(self.rules):
            m = rule_mask(X, conds) & ~done
            pred[m], which[m] = cls, k
            done |= m
        return pred, which

    def predict(self, X):
        return self.predict_trace(X)[0]

    def rule_text(self, k):
        conds, cls, why = self.rules[k]
        return f"IF {' AND '.join(cond_text(c) for c in conds)} THEN {CLASSES[cls]}   # {why}"

    def describe(self):
        lines = ["ALGORITHM 1: ORDERED IF-THEN RULE LIST (first matching rule fires)"]
        lines += [f"{k + 1:>2}. {self.rule_text(k)}" for k in range(len(self.rules))]
        lines.append(f"{len(self.rules) + 1:>2}. OTHERWISE {CLASSES[self.default]}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Algorithm 2: weighted scoring with explicit decision rules (hand-written)
# --------------------------------------------------------------------------
# Points earned by each attribute level (index = level rank, 0 = worst).
POINTS = {
    "buying": [0, 2, 4, 6],      # price counts double: the dearer the car, the fewer the points
    "maint": [0, 2, 4, 6],
    "doors": [0, 1, 2, 2],
    "persons": [0, 2, 3],
    "lug_boot": [0, 1, 2],
    "safety": [0, 2, 5],
}
# Class thresholds as a FRACTION of the maximum possible score (24 points), highest first.
THRESHOLDS = [(0.86, 3), (0.78, 2), (0.54, 1)]  # >= 86% vgood; >= 78% good; >= 54% acc; else unacc
# Veto rules: if any holds the car is unacceptable regardless of its score.
VETO_RULES = [
    ([("persons", "==", "2")], "A two-seater cannot carry a family"),
    ([("safety", "==", "low")], "Low safety is never acceptable"),
]


class WeightedScoring:
    """Score = sum of points over the six attributes. Decision rules:
       1. IF a veto rule holds THEN unacc.
       2. ELSE compare score / max_score with the thresholds (vgood, good, acc, else unacc).
    Variants (for ablation, fixed in advance): use_veto=False removes rule 1;
    equal_weights=True gives every attribute the same 0..3 scale instead of the hand-set points."""
    name = "Algorithm 2: weighted scoring"

    def __init__(self, points=None, thresholds=None, veto=None, use_veto=True, equal_weights=False, label=None):
        pts = POINTS if points is None else points
        if equal_weights:
            pts = {a: [round(3 * i / (len(LEVELS[a]) - 1), 2) for i in range(len(LEVELS[a]))] for a in COLUMNS}
        self.points, self.use_veto = pts, use_veto
        self.thresholds = THRESHOLDS if thresholds is None else thresholds
        self.veto = VETO_RULES if veto is None else veto
        self.max_score = sum(max(p) for p in self.points.values())
        if label:
            self.name = label

    def fit(self, X, y):  # nothing to learn; kept for a common interface
        return self

    def score(self, X):
        return sum(np.array(self.points[a])[X[a].to_numpy()] for a in COLUMNS)

    def predict(self, X):
        frac = self.score(X) / self.max_score
        pred = np.zeros(len(X), dtype=int)
        for thr, cls in sorted(self.thresholds, reverse=True):
            pred = np.where((pred == 0) & (frac >= thr), cls, pred)
        if self.use_veto:
            for conds, _why in self.veto:
                pred[rule_mask(X, conds)] = 0
        return pred

    def describe(self):
        lines = ["ALGORITHM 2: WEIGHTED SCORING WITH DECISION RULES", "Points per attribute level:"]
        for a in COLUMNS:
            lines.append(f"  {a:<9}" + ", ".join(f"{lv}={p:g}" for lv, p in zip(LEVELS[a], self.points[a])))
        lines.append(f"Maximum possible score = {self.max_score:g}")
        if self.use_veto:
            lines.append("Decision rule 1 (veto): IF any of these holds THEN unacc")
            lines += [f"    - {' AND '.join(cond_text(c) for c in conds)}   # {why}" for conds, why in self.veto]
        lines.append("Decision rule 2: otherwise use score as a share of the maximum:")
        for thr, cls in sorted(self.thresholds, reverse=True):
            lines.append(f"    score >= {thr:.0%} of maximum -> {CLASSES[cls]}")
        lines.append("    below that -> unacc")
        return "\n".join(lines)
