# Universal Bank — Personal Loan Propensity & Latent-Class Segmentation

An interactive Streamlit app that (1) predicts Personal Loan uptake with three tree
classifiers and (2) uses Latent Class Analysis (LCA) to find customer segments that
*resemble* loan-takers but are scored low by the classifier — the look-alikes you would
otherwise miss.

## Run

```bash
pip install -r requirements.txt
streamlit run universal_bank_app.py
```

Place `UniversalBank_with_description.xls` next to the script, or upload it via the sidebar.

## What's in the app

| Tab | Contents |
|---|---|
| **1 · Data & Feature Engineering** | Class-imbalance view, data-quality corrections, engineered-feature table |
| **2 · Classification** | Decision Tree, Random Forest, Gradient Boosted Tree — train/test accuracy, precision, recall, F1, per-model confusion matrix and feature-importance chart, and one combined ROC curve |
| **3 · Latent Class Analysis** | Slider for number of classes (k=2–8), BIC/AIC/entropy model-selection table, loan-rate-by-class bar, interactive 3-D PCA segment map, per-class descriptive profile |
| **4 · Missed Look-alikes** | Customers a chosen classifier predicts "No" who sit in high-propensity latent classes; downloadable prospect list |

## Method notes (auditable)

**Data-quality corrections** (applied to a copy; originals untouched)
- 52 records had impossible negative `Experience` (min −3). Imputed with the **median
  Experience of customers of the same Age**, then floored at 0. `Experience` tracks `Age`
  closely, so this preserves structure. *This is a defensible correction, not ground truth.*
- 1 malformed `ZIP Code` (4-digit `9307`). `ZIP Code` is dropped from modelling anyway —
  467 distinct values, not predictive, and it carries the bad value.
- `ID` dropped (identifier, not a feature).

**Engineered features** (additive): `Income_per_Family`, `CCAvg_Annual`,
`CCAvg_to_Income`, `Has_Mortgage`, `Mortgage_to_Income`.

**Classification**
- Held-out evaluation: 30% stratified test split, `random_state=42`.
- Imbalance (9.6% positive) handled with `class_weight="balanced"` (DT, RF) and inverse-
  prevalence `sample_weight` (GBT, which has no `class_weight`).
- Precision/Recall/F1 reported for the **positive class** (loan accepted) — the targeting class.

**Latent Class Analysis** — `stepmix` (Morin et al.), `measurement="categorical"`.
- Continuous columns (`Age`, `Experience`, `Income`, `CCAvg`, `Mortgage`) binned into
  ordered domain bands; categorical columns used as-is. Binning thresholds are shown in Tab 1/3.
- Model selection via BIC/AIC (lower better) and normalised relative entropy (higher = cleaner
  class separation). LCA is unsupervised — `Personal Loan` is **not** used to fit it; the loan
  rate is overlaid afterward to interpret each class.
- The 3-D scatter is PCA on the one-hot categorical matrix, **for visualisation only**; the LCA
  fit uses the full categorical structure, not the PCA projection.

## Caveats
- Tab 4 is a **prospecting aid, not a credit decision**. The look-alike list mixes true acceptors
  the model missed with genuine non-acceptors who merely resemble them. Treat class membership as
  a propensity signal for review, not a verdict.
- LCA fits are stochastic; `random_state=42` and `n_init=5` are set for reproducibility, but BIC
  can shift slightly with different seeds/initialisations.
- Random Forest reports near-perfect training accuracy (1.000), which is expected for an
  unconstrained forest and not by itself a problem — judge generalisation by the **test** metrics.
