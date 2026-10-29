"""
rb_car.py - Rule-based AI components for the Car Evaluation dataset
===================================================================
CPU5006 Artificial Intelligence, Assessment 1 (Rule-Based AI Experiments).

Two rule-based algorithms (plus two simple baselines):

  1. HierarchicalExpertSystem - hand-built forward-chaining production system that
                     mirrors the documented concept hierarchy (PRICE, COMFORT, TECH -> CAR).
                     It is NOT trained on data.
  2. CN2           - ordered rule-list *induction* (Clark & Niblett, 1989) with the
                     Laplace heuristic (Clark & Boswell, 1991), generalised here to
                     multi-class problems and ordinal selectors. Written from scratch.
  3. OneR          - Holte's (1993) 1R baseline.
  4. MajorityClass - trivial baseline.

AI TOOL DISCLOSURE (required by the assessment brief):
  This code was developed with the assistance of Claude (Anthropic, model
  Sonnet 5.5) in October 2026. Claude produced the first draft of the
  implementation and experiment harness; I reviewed, ran and adapted it. Update
  this note with exactly what you changed/checked before submitting.

Only numpy / pandas / scipy are used by the algorithms (no ML library for the models).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2

COLUMNS = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
# Ordinal level order for every attribute (rank 0 = least desirable for the buyer).
LEVELS = {
    "buying": ["vhigh", "high", "med", "low"],       # low price is best
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
    np.add.at(m, (y, pred), 1)
    return m


def evaluate(y, pred):
    m = confusion(np.asarray(y), np.asarray(pred))
    n = m.sum()
    tp = np.diag(m).astype(float)
    rec = np.divide(tp, m.sum(1), out=np.zeros(K), where=m.sum(1) > 0)
    prec = np.divide(tp, m.sum(0), out=np.zeros(K), where=m.sum(0) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(K), where=(prec + rec) > 0)
    po = tp.sum() / n
    pe = (m.sum(0) * m.sum(1)).sum() / n ** 2
    out = {"accuracy": po, "macro_f1": f1.mean(), "kappa": (po - pe) / (1 - pe) if pe < 1 else 0.0,
           "ordinal_mae": float(np.abs(np.asarray(y) - np.asarray(pred)).mean())}
    for k, c in enumerate(CLASSES):
        out[f"recall_{c}"], out[f"precision_{c}"] = rec[k], prec[k]
    return out


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
class MajorityClass:
    name = "Majority class"

    def fit(self, X, y):
        self.cls = int(np.bincount(y, minlength=K).argmax())
        return self

    def predict(self, X):
        return np.full(len(X), self.cls)

    def n_rules(self):
        return 1

    def avg_len(self):
        return 0.0


class OneR:
    """Holte (1993): one attribute, one rule per value; keep the attribute with fewest training errors."""
    name = "1R"

    def fit(self, X, y):
        best = None
        for c in COLUMNS:
            table, err = {}, 0
            for v in np.unique(X[c]):
                cnt = np.bincount(y[X[c].to_numpy() == v], minlength=K)
                table[v] = int(cnt.argmax())
                err += cnt.sum() - cnt.max()
            if best is None or err < best[0]:
                best = (err, c, table)
        _, self.attr, self.table = best
        self.default = int(np.bincount(y, minlength=K).argmax())
        return self

    def predict(self, X):
        return np.array([self.table.get(v, self.default) for v in X[self.attr]])

    def n_rules(self):
        return len(self.table)

    def avg_len(self):
        return 1.0


# --------------------------------------------------------------------------
# Algorithm 1: hierarchical forward-chaining expert system
# --------------------------------------------------------------------------
class HierarchicalExpertSystem:
    """Forward-chaining production system with three inference layers.

    Layer 1 derives intermediate concepts from raw attributes
        PRICE   (from buying, maint)               -> cheap | fair | expensive
        COMFORT (from persons, doors, lug_boot)    -> unsuitable | low | medium | high
    Layer 2 derives
        TECH    (from COMFORT, safety)             -> unsafe | poor | adequate | good | excellent
    Layer 3 maps (PRICE, TECH) to the final class using an ORDERED rule list
    (first matching rule fires; conflict resolution by rule order).

    The rules follow the concept hierarchy documented with the dataset and general
    knowledge about cars. They were written ONCE and were not tuned on any results.
    """
    name = "Hierarchical expert system"

    # Layer 1/2 are small decision tables expressed as functions (documented in describe()).
    @staticmethod
    def derive(r):
        f = {}
        price_pts = r["buying"] + r["maint"]  # 0 (vhigh+vhigh) .. 6 (low+low)
        f["price"] = "cheap" if price_pts >= 5 else ("fair" if price_pts >= 3 else "expensive")
        if r["persons"] == 0:
            f["comfort"] = "unsuitable"
        else:
            pts = r["persons"] + r["lug_boot"] + min(r["doors"], 2)  # 1..3 + 0..2 + 0..2
            f["comfort"] = "high" if pts >= 6 else ("medium" if pts >= 4 else "low")
        if r["safety"] == 0 or f["comfort"] == "unsuitable":
            f["tech"] = "unsafe"
        elif r["safety"] == 2 and f["comfort"] == "high":
            f["tech"] = "excellent"
        elif r["safety"] == 2 and f["comfort"] == "medium":
            f["tech"] = "good"
        elif r["safety"] >= 1 and f["comfort"] in ("medium", "high"):
            f["tech"] = "adequate"
        else:
            f["tech"] = "poor"
        return f

    # Layer 3: ordered rule list over (price, tech)
    FINAL = [
        (lambda f: f["tech"] in ("unsafe", "poor"), 0, "IF tech is unsafe/poor THEN unacc"),
        (lambda f: f["price"] == "expensive", 0, "IF price is expensive THEN unacc"),
        (lambda f: f["price"] == "cheap" and f["tech"] == "excellent", 3, "IF price cheap AND tech excellent THEN vgood"),
        (lambda f: f["price"] == "cheap" and f["tech"] == "good", 2, "IF price cheap AND tech good THEN good"),
        (lambda f: f["price"] == "fair" and f["tech"] == "excellent", 2, "IF price fair AND tech excellent THEN good"),
        (lambda f: True, 1, "ELSE acc"),
    ]

    def fit(self, X, y):  # nothing to learn; kept for a common interface
        return self

    def infer(self, row):
        f = self.derive(row)
        for k, (cond, cls, text) in enumerate(self.FINAL):
            if cond(f):
                return cls, f, text
        raise RuntimeError("unreachable")

    def predict(self, X):
        return np.array([self.infer(r)[0] for _, r in X.iterrows()])

    def explain(self, row_series):
        cls, f, text = self.infer(row_series)
        return f"price={f['price']}, comfort={f['comfort']}, tech={f['tech']} -> {text}"

    def n_rules(self):
        # 3 price + 4 comfort + 5 tech outcomes (layers 1-2) + 6 final rules (layer 3)
        return 3 + 4 + 5 + len(self.FINAL)

    def avg_len(self):
        return 2.0

    def describe(self):
        return "\n".join([
            "HIERARCHICAL EXPERT SYSTEM",
            "Layer 1  price   : buying+maint points >=5 cheap; 3-4 fair; <=2 expensive (vhigh=0 ... low=3 each)",
            "Layer 1  comfort : persons=2 -> unsuitable; else points = persons rank (4 -> 1, more -> 2)",
            "                   + lug_boot rank (small 0, med 1, big 2) + min(doors rank, 2) (2 doors 0, 3 doors 1, 4+ 2);",
            "                   points >= 6 high; 4-5 medium; otherwise low",
            "Layer 2  tech    : safety low or comfort unsuitable -> unsafe; high safety & high comfort -> excellent;",
            "                   high safety & medium comfort -> good; safety>=med & comfort>=medium -> adequate; else poor",
            "Layer 3  (ordered, first match):",
            *[f"   {i + 1}. {t}" for i, (_, _, t) in enumerate(self.FINAL)],
        ])


# --------------------------------------------------------------------------
# Algorithm 2: multi-class CN2 ordered rule-list induction
# --------------------------------------------------------------------------
class CN2:
    """CN2 (Clark & Niblett 1989) with Laplace accuracy (Clark & Boswell 1991).

    Sequential covering: (1) beam-search the best IF-THEN rule on the examples not
    yet covered, (2) append it to an ORDERED rule list, (3) remove the covered
    examples. A rule is accepted only if it covers >= `min_cov` examples and passes a
    likelihood-ratio significance test against the class distribution of the
    remaining data (chi-squared, K-1 d.f., alpha = 0.05). The default rule predicts
    the majority class of whatever remains.

    Selectors: 'attr == level'; with ordinal=True also 'attr <= level' and
    'attr >= level' (the attributes are ordinal), which gives much shorter rules.
    """
    name = "CN2 (induced ordered rule list)"

    def __init__(self, beam=5, max_len=4, min_cov=5, ordinal=True, alpha=0.05):
        self.beam, self.max_len, self.min_cov = beam, max_len, min_cov
        self.ordinal, self.sig = ordinal, chi2.ppf(1 - alpha, K - 1)

    def _selectors(self):
        sel = []
        for c in COLUMNS:
            L = len(LEVELS[c])
            sel += [(c, "==", v) for v in range(L)]
            if self.ordinal:
                sel += [(c, "<=", v) for v in range(L - 1)] + [(c, ">=", v) for v in range(1, L)]
        return sel

    @staticmethod
    def _mask(X, s):
        c, op, v = s
        col = X[c].to_numpy()
        return col == v if op == "==" else (col <= v if op == "<=" else col >= v)

    def _matrix(self, X):
        return np.column_stack([self._mask(X, s) for s in self.selectors])

    def _lrt(self, cnt, prior):
        n = cnt.sum()
        exp = n * prior
        ok = (cnt > 0) & (exp > 0)
        return 2 * float((cnt[ok] * np.log(cnt[ok] / exp[ok])).sum())

    def _allowed(self, rule, j):
        c, op, _ = self.selectors[j]
        same = [self.selectors[i] for i in rule if self.selectors[i][0] == c]
        if not same:
            return True
        if op == "==" or any(u[1] == "==" for u in same):
            return False
        return len(same) < 2 and same[0][1] != op  # at most an interval (>= and <=)

    def _learn_one(self, S, y, active):
        prior = np.bincount(y[active], minlength=K) / active.sum()
        best, best_q = None, -1.0
        beam = [()]
        for _ in range(self.max_len):
            cand = {}
            for rule in beam:
                cov = active.copy()
                for j in rule:
                    cov &= S[:, j]
                for j in range(S.shape[1]):
                    if j in rule or not self._allowed(rule, j):
                        continue
                    new = tuple(sorted(rule + (j,)))
                    if new in cand:
                        continue
                    m = cov & S[:, j]
                    n = int(m.sum())
                    if n < self.min_cov:
                        continue
                    cnt = np.bincount(y[m], minlength=K)
                    cand[new] = ((cnt.max() + 1) / (n + K), cnt)
            if not cand:
                break
            ranked = sorted(cand.items(), key=lambda kv: kv[1][0], reverse=True)
            beam = [r for r, _ in ranked[: self.beam]]
            for r, (q, cnt) in ranked:
                if q > best_q and self._lrt(cnt, prior) >= self.sig:
                    best, best_q = (r, cnt), q
        return best

    def fit(self, X, y):
        y = np.asarray(y)
        self.selectors = self._selectors()
        S = self._matrix(X)
        active = np.ones(len(y), dtype=bool)
        self.rules = []  # (selector list, class, class counts)
        while active.sum() >= self.min_cov:
            found = self._learn_one(S, y, active)
            if found is None:
                break
            idx, cnt = found
            cov = active.copy()
            for j in idx:
                cov &= S[:, j]
            self.rules.append(([self.selectors[j] for j in idx], int(cnt.argmax()), cnt))
            active &= ~cov
        rem = y[active] if active.any() else y
        self.default_cls = int(np.bincount(rem, minlength=K).argmax())
        return self

    def _fire(self, X):
        n = len(X)
        pred = np.full(n, self.default_cls)
        which = np.full(n, -1)
        done = np.zeros(n, dtype=bool)
        for k, (conds, cls, _cnt) in enumerate(self.rules):
            m = np.ones(n, dtype=bool)
            for s in conds:
                m &= self._mask(X, s)
            m &= ~done
            pred[m], which[m] = cls, k
            done |= m
        return pred, which

    def predict(self, X):
        return self._fire(X)[0]

    def n_rules(self):
        return len(self.rules) + 1

    def avg_len(self):
        return float(np.mean([len(r[0]) for r in self.rules])) if self.rules else 0.0

    @staticmethod
    def _cond_text(s):
        c, op, v = s
        return f"{c} {op} {LEVELS[c][v]}"

    def rule_text(self, k):
        conds, cls, cnt = self.rules[k]
        return (f"IF {' AND '.join(self._cond_text(s) for s in conds)} THEN {CLASSES[cls]}"
                f"  [covers {cnt.sum()}, {cnt[cls]} correct]")

    def describe(self):
        lines = [f"CN2 ORDERED RULE LIST ({len(self.rules)} rules + default)"]
        lines += [f"{k + 1:>3}. {self.rule_text(k)}" for k in range(len(self.rules))]
        lines.append(f"{len(self.rules) + 1:>3}. ELSE {CLASSES[self.default_cls]}")
        return "\n".join(lines)
