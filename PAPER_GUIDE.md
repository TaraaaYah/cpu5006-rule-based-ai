# Paper guide – CPU5006 Assessment 1 (Car Evaluation, two hand-written rule systems)

This is a blueprint and evidence pack, **not** the paper. The marking criteria reward your own critical analysis and
penalise unadapted AI-generated text, so the writing needs to be yours. Every number below comes from
`python run_experiments.py` on the held-out test cars.

## 1. Research question and hypotheses
> Does a points-based weighted scoring system classify car acceptability better than a straightforward ordered
> IF-THEN rule list, and what does each approach gain or lose in accuracy, rare-class performance and interpretability?

- **H1:** weighted scoring will not beat the IF-THEN list on rare classes (good, vgood). *Supported on macro-F1.*
- **H2:** scoring needs explicit veto rules, because a high score can compensate for a fatal flaw. *Strongly supported.*

## 2. Method in brief (for your methodology section)
- 1,728 cars; split 30% development (518) / 70% test (1,210), stratified, seed 42 (`split.py`).
- Rules were refined on the development cars only (`results/rule_development_log.md`), then frozen. The test cars were
  used once.
- Algorithm 1: ten ordered IF-THEN rules, first match wins (knock-outs first, then best classes, default "acc").
- Algorithm 2: points per attribute (max 24), two veto rules, class thresholds at 54% / 78% / 86% of the maximum.
- Metrics: accuracy, macro-F1, Cohen's kappa, ordinal MAE, per-class recall and precision. 95% bootstrap CIs
  (2,000 resamples); exact McNemar test for the paired accuracy comparison.

## 3. Results on the 1,210 test cars (95% CI in brackets)

| System | Accuracy | Macro-F1 | Kappa | Ordinal MAE |
|---|---|---|---|---|
| Always "unacc" | 0.700 [0.674, 0.726] | 0.206 | 0.000 | 0.416 |
| Algorithm 1: IF-THEN | **0.909** [0.893, 0.926] | **0.796** [0.754, 0.837] | **0.809** [0.777, 0.843] | **0.098** [0.079, 0.116] |
| Algorithm 2: scoring | 0.895 [0.878, 0.912] | 0.689 [0.633, 0.743] | 0.769 [0.735, 0.806] | 0.112 [0.092, 0.131] |

Recall per class (unacc / acc / good / vgood): Algorithm 1 = 0.93 / 0.90 / 0.46 / 1.00; Algorithm 2 = 0.96 / 0.86 / 0.42 / 0.44.

- **Accuracy:** no significant difference (difference +0.014, CI -0.007 to +0.035; exact McNemar p = 0.21).
- **Macro-F1:** Algorithm 1 is better by 0.107 (CI 0.038 to 0.172).
- **Algorithm 2 ablation** (`results/algorithm2_ablation.csv`): without veto rules accuracy falls from 0.895 to 0.721 and
  macro-F1 from 0.689 to 0.586; with equal attribute weights macro-F1 falls to 0.561.
- **Development vs test** (macro-F1): Algorithm 1 0.846 -> 0.796; Algorithm 2 0.700 -> 0.689.

## 4. Things worth discussing (this is where the marks are)
1. **Accuracy hides the real difference.** The two systems are statistically tied on accuracy but differ a lot on rare classes:
   Algorithm 2 finds only 44% of very-good cars against 100% for Algorithm 1.
2. **A score is compensatory; some rules are not.** Removing the veto rules costs 17 points of accuracy. This is
   the main evidence for H2 and connects to your literature review (DEX is built from non-linear decision tables).
3. **Weights matter.** Equal weights drop macro-F1 from 0.69 to 0.56, so the knowledge in the points table is doing work.
4. **"good" is hard for both** (recall 0.46 and 0.42; only 48 test cars). Most errors are between neighbouring classes
   (ordinal MAE about 0.1), so the systems are rarely wildly wrong.
5. **Weak rules** (`results/algorithm1_per_rule_test.csv`): rule 9 fires 17 times and is right only 35% of the time; the
   catch-all default fires 314 times and mislabels 53 unacceptable cars as "acc".
6. **Overfitting to the development set.** Algorithm 1, refined by hand over three rounds, loses 0.05 macro-F1 on the test
   cars; the less flexible Algorithm 2 loses 0.01. That is a known risk of manual rule tuning.
7. **Interpretability and knowledge acquisition.** Ten readable rules are easy to audit; a points table is compact but a
   "good" decision can only be explained as "the total reached 19 points". Both need a person to maintain them.

## 5. Limitations you should state honestly
- **The rules were drafted by an AI assistant,** not elicited from a human expert, and that assistant had earlier seen
  results for the whole dataset. The development/test split limits further tuning but does not remove that influence.
- Algorithm 2's thresholds were chosen by a small grid search on the development cars; Algorithm 1 was refined by hand.
- One split only. The confidence intervals capture test-sampling variation, not variation from the rule-writing process.
- Rare classes are small (good 48, vgood 46 test cars), so their recall estimates are wide.
- The dataset is complete and noise-free, so these results say little about noisy real-world data.

## 6. Section plan (2,000-3,000 words; aim for about 2,500-2,800)

| Section | Words | Marks | What to include |
|---|---|---|---|
| GitHub link | – | – | Repo URL at the very top |
| Introduction | 300 | 25% (with lit review) | Context, why transparent rules matter, your question and hypotheses |
| Literature review | 600 | 25% | Your draft plus: 10-15 sources, critical comparison, how rule-based inference works, multi-attribute decision models, the research gap and question |
| Methodology | 700 | 25% | Dataset and class imbalance; both algorithms with design decisions; development/test protocol; metrics and why; statistics |
| Results and discussion | 900 | 25% | Tables 1-2, Figs 1-4, points 1-7 above; challenge your own results |
| Conclusion | 300 | 15% | Answer the question; key findings; limitations; future work |
| References | – | – | Harvard, UK English |

Figures: `fig1_performance.png`, `fig2_confusion.png`, `fig3_dev_vs_test.png`, `fig4_algorithm2_ablation.png`.
Number and caption each one and refer to it in the text. Type tables into Word rather than pasting screenshots.

## 7. Fixes for your literature review draft
- Dataset citation: donated by Bohanec and Zupan in **1997**; the 1988 paper is by **Bohanec and Rajkovic**.
- Check Quinlan (1987) against the original publication details.
- Cite the "transparency" claim and qualify it; add the research gap and your question at the end of 2.4.

## 8. Reading list (verify every detail before citing)
- Bohanec, M. and Rajkovic, V. (1988) Knowledge acquisition and explanation for multi-attribute decision making. *8th Intl Workshop on Expert Systems and their Applications*, Avignon (cited in `car.names`).
- Zupan, B., Bohanec, M., Bratko, I. and Demsar, J. (1997) Machine learning by function decomposition. *ICML-97* (cited in `car.names`).
- Buchanan, B.G. and Shortliffe, E.H. (eds) (1984) *Rule-Based Expert Systems: The MYCIN Experiments*. Addison-Wesley.
- Quinlan, J.R. (1987) Generating production rules from decision trees. *IJCAI-87*.
- Clark, P. and Niblett, T. (1989) The CN2 induction algorithm. *Machine Learning*, 3(4).
- Rudin, C. (2019) Stop explaining black box models for high-stakes decisions. *Nature Machine Intelligence*, 1.
- Cohen, J. (1960) A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1).
- Efron, B. and Tibshirani, R. (1993) *An Introduction to the Bootstrap*. Chapman and Hall.
- UCI Machine Learning Repository: Car Evaluation dataset (1997).

## 9. Before you submit
1. Reword the AI-use note at the top of `rb_car.py`, `split.py`, `develop_rules.py` and `run_experiments.py` in your own words.
2. Be able to explain every rule and the points table. Run `python run_experiments.py` on your own PC.
3. Public repo link at the top of the paper; submit Word or PDF through Turnitin in Minerva by 15 December 2026.
