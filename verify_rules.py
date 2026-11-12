"""verify_rules.py - independent check that the rule code does what the paper describes.

It re-implements both algorithms in a few plain lines, directly from the tables in the paper,
and compares them with rb_car.py on all 1,728 cars. It also checks the split and prints
five worked examples to trace by hand.

AI TOOL DISCLOSURE: this script was drafted by Claude (Anthropic, Sonnet 5.5), November 2026.
Run it with: python verify_rules.py
"""
import numpy as np
import pandas as pd
from rb_car import load_data, COLUMNS, CLASSES, LEVELS, IfThenRuleList, WeightedScoring
from split import dev_test_indices

X, y = load_data("data/car.data")
raw = pd.read_csv("data/car.data", header=None, names=COLUMNS + ["class"], dtype=str)
a1, a2 = IfThenRuleList(), WeightedScoring()
p1, which = a1.predict_trace(X)
p2, score = a2.predict(X), a2.score(X)

dev, test = dev_test_indices(y)
print("Split: no overlap =", not set(dev) & set(test), "| sizes", len(dev), len(test),
      "| covers all cars =", len(set(dev) | set(test)) == len(y))


def rank(c, v):
    return LEVELS[c].index(v)


def algo1(r):  # written from Table 1 of the paper
    b, m, p, l, s = (rank("buying", r["buying"]), rank("maint", r["maint"]), rank("persons", r["persons"]),
                     rank("lug_boot", r["lug_boot"]), rank("safety", r["safety"]))
    if p == 0 or s == 0: return 0
    if b == 0 and m <= 1: return 0
    if m == 0 and b <= 1: return 0
    if s == 1 and b == 1 and l <= 1: return 0
    if s == 2 and l >= 1 and b >= 2 and m >= 2: return 3
    if s == 2 and b == 3 and m == 1 and l >= 1: return 3
    if s == 2 and b >= 2 and m >= 2: return 2
    if s == 1 and b == 3 and m == 3: return 2
    return 1


PTS = {"buying": [0, 2, 4, 6], "maint": [0, 2, 4, 6], "doors": [0, 1, 2, 2],
       "persons": [0, 2, 3], "lug_boot": [0, 1, 2], "safety": [0, 2, 5]}


def algo2(r):  # written from Table 2 of the paper
    if r["persons"] == "2" or r["safety"] == "low": return 0
    f = sum(PTS[c][rank(c, r[c])] for c in COLUMNS) / 24
    return 3 if f >= .86 else 2 if f >= .78 else 1 if f >= .54 else 0


recs = [{c: raw.iloc[i][c].replace("-", "") for c in COLUMNS} for i in range(len(raw))]
print("Algorithm 1 re-implementation matches rb_car.py on all cars:", (np.array([algo1(r) for r in recs]) == p1).all())
print("Algorithm 2 re-implementation matches rb_car.py on all cars:", (np.array([algo2(r) for r in recs]) == p2).all())
print("\nWorked examples (test cars): trace each by hand against Tables 1 and 2")
for i in (test[0], test[100], test[400], test[800], test[1100]):
    rule = which[i] + 1 if which[i] >= 0 else len(a1.rules) + 1
    print(list(raw.iloc[i].values[:6]), "true:", raw.iloc[i]["class"], "| Algorithm 1: rule", rule, "->", CLASSES[p1[i]],
          "| Algorithm 2: score", int(score[i]), "/ 24 ->", CLASSES[p2[i]])
