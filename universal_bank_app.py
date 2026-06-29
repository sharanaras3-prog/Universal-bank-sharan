"""
Universal Bank — Personal Loan Propensity & Latent-Class Segmentation
=====================================================================
Interactive app combining:
  (1) Supervised classification (Decision Tree, Random Forest, Gradient Boosted Tree)
      to predict Personal Loan uptake, and
  (2) Latent Class Analysis (LCA) on a categorical recoding of the data to surface
      customer segments that "look like" loan-takers but may be missed by the
      classifier (i.e. high-propensity look-alikes among predicted negatives).

Run:
    pip install -r requirements.txt
    streamlit run universal_bank_app.py

Notes on method (auditable):
  - Data-quality corrections and feature engineering are documented in the
    "Data & Feature Engineering" tab and applied non-destructively (originals kept).
  - All classification metrics are reported on a held-out, stratified 30% test split
    (random_state=42). Imbalance (9.6% positive) is handled with class weighting.
  - LCA is fit with stepmix (Morin et al.) on integer-coded categorical indicators.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Universal Bank — Loan Propensity & Segmentation",
                   layout="wide", initial_sidebar_state="expanded")

DATA_PATH_DEFAULT = "UniversalBank_with_description.xls"
SEED = 42
TEST_SIZE = 0.30

# Continuous columns binned for the LCA (thresholds documented in the UI).
CONTINUOUS_FOR_LCA = ["Age", "Experience", "Income", "CCAvg", "Mortgage"]


# ===========================================================================
# 1. DATA LOADING + QUALITY FIXES + FEATURE ENGINEERING
# ===========================================================================
@st.cache_data(show_spinner=False)
def load_raw(path):
    df = pd.read_excel(path, engine="xlrd", sheet_name="Data")
    return df


@st.cache_data(show_spinner=False)
def prepare_data(path):
    """Return (raw, model_df, dq_report). model_df has DQ fixes + engineered features."""
    df = load_raw(path)
    d = df.copy()
    report = {}

    # --- DQ fix 1: negative Experience (known dataset error) -------------
    neg_mask = d["Experience"] < 0
    report["neg_experience"] = int(neg_mask.sum())
    age_median = d.loc[~neg_mask].groupby("Age")["Experience"].median()
    d.loc[neg_mask, "Experience"] = d.loc[neg_mask, "Age"].map(age_median)
    d["Experience"] = d["Experience"].fillna(d["Experience"].median()).clip(lower=0)

    # --- DQ fix 2: malformed ZIP (one 4-digit value); ZIP dropped from model
    report["bad_zip"] = int((df["ZIP Code"].astype(str).str.len() != 5).sum())

    # --- Drop non-predictive identifiers ---------------------------------
    d = d.drop(columns=["ID", "ZIP Code"])

    # --- Feature engineering (additive; originals retained) --------------
    d["Income_per_Family"]  = d["Income"] / d["Family"]
    d["CCAvg_Annual"]       = d["CCAvg"] * 12          # CCAvg is $000s/month
    d["CCAvg_to_Income"]    = (d["CCAvg_Annual"] / d["Income"]).replace([np.inf, -np.inf], 0)
    d["Has_Mortgage"]       = (d["Mortgage"] > 0).astype(int)
    d["Mortgage_to_Income"] = (d["Mortgage"] / d["Income"]).replace([np.inf, -np.inf], 0)

    return df, d, report


@st.cache_data(show_spinner=False)
def build_categorical(path):
    """Recode everything to categorical for the LCA. Returns the labelled
    (human-readable) categorical frame plus the integer-coded matrix."""
    _, d, _ = prepare_data(path)
    cat = pd.DataFrame(index=d.index)

    # Binning thresholds — chosen on domain meaning + distribution, not arbitrary.
    cat["Age_grp"]    = pd.cut(d["Age"], [22, 35, 45, 55, 70],
                               labels=["≤35", "36–45", "46–55", "56+"])
    cat["Exp_grp"]    = pd.cut(d["Experience"], [-1, 5, 15, 25, 50],
                               labels=["0–5", "6–15", "16–25", "26+"])
    cat["Income_grp"] = pd.cut(d["Income"], [0, 50, 100, 150, 250],
                               labels=["Low (≤50)", "Mid (51–100)", "High (101–150)", "VHigh (151+)"])
    cat["CCAvg_grp"]  = pd.cut(d["CCAvg"], [-0.01, 1, 3, 11],
                               labels=["Low (≤1)", "Med (1–3)", "High (3+)"])
    cat["Mort_grp"]   = pd.cut(d["Mortgage"], [-1, 0, 150, 700],
                               labels=["None", "Low (1–150)", "High (150+)"])
    cat["Family"]     = d["Family"].astype(int).astype(str)
    cat["Education"]  = d["Education"].map({1: "Undergrad", 2: "Graduate", 3: "Advanced"})
    cat["Securities"] = d["Securities Account"].map({0: "No", 1: "Yes"})
    cat["CD"]         = d["CD Account"].map({0: "No", 1: "Yes"})
    cat["Online"]     = d["Online"].map({0: "No", 1: "Yes"})
    cat["CreditCard"] = d["CreditCard"].map({0: "No", 1: "Yes"})

    codes = cat.apply(lambda c: LabelEncoder().fit_transform(c.astype(str)))
    return cat, codes


# ===========================================================================
# 2. CLASSIFICATION
# ===========================================================================
@st.cache_resource(show_spinner=False)
def train_classifiers(path):
    _, d, _ = prepare_data(path)
    y = d["Personal Loan"]
    X = d.drop(columns=["Personal Loan"])
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "Gradient Boosted Tree": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, random_state=SEED),
    }

    results = {}
    for name, m in models.items():
        # GBT has no class_weight; weight positives by inverse prevalence.
        if name == "Gradient Boosted Tree":
            w = ytr.map({0: 1.0, 1: (ytr == 0).sum() / max((ytr == 1).sum(), 1)})
            m.fit(Xtr, ytr, sample_weight=w)
        else:
            m.fit(Xtr, ytr)

        ptr = m.predict(Xtr)
        pte = m.predict(Xte)
        proba = m.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, proba)
        results[name] = {
            "model": m,
            "train_acc": accuracy_score(ytr, ptr),
            "test_acc": accuracy_score(yte, pte),
            "precision": precision_score(yte, pte, zero_division=0),
            "recall": recall_score(yte, pte, zero_division=0),
            "f1": f1_score(yte, pte, zero_division=0),
            "cm": confusion_matrix(yte, pte),
            "fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr),
            "importances": pd.Series(m.feature_importances_, index=X.columns).sort_values(),
            "proba_all": m.predict_proba(X)[:, 1],
            "pred_all": m.predict(X),
        }
    return results, list(X.columns)


# ===========================================================================
# 3. LATENT CLASS ANALYSIS
# ===========================================================================
@st.cache_resource(show_spinner=False)
def fit_lca(path, k, n_init=5, max_iter=300):
    from stepmix.stepmix import StepMix
    _, codes = build_categorical(path)
    m = StepMix(n_components=k, measurement="categorical",
                random_state=SEED, n_init=n_init, max_iter=max_iter, verbose=0)
    m.fit(codes)
    labels = m.predict(codes)
    post = m.predict_proba(codes)            # posterior class probabilities
    try:
        bic = m.bic(codes); aic = m.aic(codes)
    except Exception:
        bic = aic = np.nan
    # Normalised entropy (1 = perfectly separated classes)
    eps = 1e-12
    ent = -np.sum(post * np.log(post + eps), axis=1)
    rel_entropy = 1 - ent.sum() / (len(post) * np.log(k)) if k > 1 else 1.0
    return labels, post, bic, aic, rel_entropy


@st.cache_data(show_spinner=False)
def lca_scan(path, k_min=2, k_max=8):
    """BIC/AIC/entropy across a range of k for model selection."""
    rows = []
    for k in range(k_min, k_max + 1):
        _, _, bic, aic, ent = fit_lca(path, k)
        rows.append({"k": k, "BIC": bic, "AIC": aic, "Rel. entropy": ent})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def pca_3d(path):
    """One-hot the categorical recoding, then PCA -> 3 components for plotting."""
    cat, _ = build_categorical(path)
    dummies = pd.get_dummies(cat.astype(str))
    p = PCA(n_components=3, random_state=SEED)
    comps = p.fit_transform(dummies.values.astype(float))
    return comps, p.explained_variance_ratio_


# ===========================================================================
# UI
# ===========================================================================
st.title("Universal Bank — Personal Loan Propensity & Latent-Class Segmentation")

with st.sidebar:
    st.header("Data source")
    up = st.file_uploader("UniversalBank .xls (optional — defaults to bundled file)",
                          type=["xls", "xlsx"])
    path = up if up is not None else DATA_PATH_DEFAULT
    st.caption("Target: **Personal Loan** (1 = accepted the loan offer).")

try:
    raw, model_df, dq = prepare_data(path)
except Exception as e:
    st.error(f"Could not load data from `{path}`.\n\n{e}")
    st.stop()

base_rate = raw["Personal Loan"].mean()

tab_data, tab_clf, tab_lca, tab_missed = st.tabs(
    ["1 · Data & Feature Engineering",
     "2 · Classification",
     "3 · Latent Class Analysis",
     "4 · Missed Look-alikes"])

# ---------------------------------------------------------------------------
# TAB 1 — DATA
# ---------------------------------------------------------------------------
with tab_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(raw):,}")
    c2.metric("Accepted loan", f"{int(raw['Personal Loan'].sum()):,}")
    c3.metric("Base acceptance rate", f"{base_rate:.1%}")
    c4.metric("Class imbalance", f"1 : {round((1-base_rate)/base_rate)}")

    st.subheader("Class imbalance")
    vc = raw["Personal Loan"].map({0: "No loan", 1: "Accepted loan"}).value_counts()
    st.plotly_chart(px.bar(x=vc.index, y=vc.values,
                           labels={"x": "", "y": "Customers"},
                           color=vc.index,
                           color_discrete_map={"No loan": "#9aa5b1", "Accepted loan": "#2563eb"}),
                    use_container_width=True)
    st.info(f"Only **{base_rate:.1%}** of customers accepted — a strong imbalance. "
            "Classifiers below use class weighting so the minority is not ignored; "
            "the LCA tab then finds non-acceptor segments that resemble acceptors.")

    st.subheader("Data-quality corrections (applied non-destructively)")
    st.markdown(f"""
| Issue | Detected | Treatment |
|---|---|---|
| **Negative `Experience`** (impossible values) | **{dq['neg_experience']} records** (min −3 years) | Imputed with the **median Experience of customers of the same Age**, then floored at 0. Experience tracks Age closely, so this preserves structure. |
| **Malformed `ZIP Code`** | **{dq['bad_zip']} record** (4-digit value 9307) | `ZIP Code` **dropped** from modelling — high cardinality (467 values), not predictive, and contains the bad value. |
| **`ID` column** | identifier | Dropped — not a feature. |
""")
    st.caption("⚠️ The Experience imputation is a defensible correction, not ground truth — "
               "if the source system can supply corrected values, prefer those.")

    st.subheader("Engineered features (originals retained)")
    st.markdown("""
| Feature | Definition | Rationale |
|---|---|---|
| `Income_per_Family` | Income ÷ Family size | Per-capita affordability |
| `CCAvg_Annual` | CCAvg × 12 | Annualised card spend ($000s) |
| `CCAvg_to_Income` | CCAvg_Annual ÷ Income | Spending intensity relative to income |
| `Has_Mortgage` | Mortgage > 0 | 69% of customers carry no mortgage — a clean binary signal |
| `Mortgage_to_Income` | Mortgage ÷ Income | Leverage proxy |
""")
    with st.expander("Preview engineered modelling table"):
        st.dataframe(model_df.head(20), use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2 — CLASSIFICATION
# ---------------------------------------------------------------------------
with tab_clf:
    with st.spinner("Training Decision Tree, Random Forest, Gradient Boosted Tree…"):
        results, feat_cols = train_classifiers(path)

    st.subheader("Performance on held-out test set (30%, stratified)")
    summary = pd.DataFrame({
        n: {"Train acc.": r["train_acc"], "Test acc.": r["test_acc"],
            "Precision": r["precision"], "Recall": r["recall"],
            "F1": r["f1"], "ROC-AUC": r["auc"]}
        for n, r in results.items()}).T
    st.dataframe(summary.style.format("{:.3f}"), use_container_width=True)
    st.caption("Precision / Recall / F1 are for the **positive class (loan accepted)** — "
               "the class that matters for targeting. Recall = share of true acceptors caught.")

    st.subheader("Combined ROC curve")
    fig = go.Figure()
    for n, r in results.items():
        fig.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], mode="lines",
                                 name=f"{n} (AUC={r['auc']:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color="gray"), name="Chance"))
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                      legend=dict(x=0.5, y=0.05), height=480)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Per-model detail")
    pick = st.radio("Model", list(results.keys()), horizontal=True)
    r = results[pick]
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Confusion matrix** (test set)")
        cm = r["cm"]
        cmfig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                          labels=dict(x="Predicted", y="Actual", color="Count"),
                          x=["No loan", "Loan"], y=["No loan", "Loan"])
        cmfig.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(cmfig, use_container_width=True)
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"True acceptors **missed** by this model (false negatives): **{fn}**. "
                   f"These — and their look-alikes — are the focus of Tab 4.")
    with cc2:
        st.markdown("**Feature importance**")
        imp = r["importances"]
        impfig = px.bar(x=imp.values, y=imp.index, orientation="h",
                        labels={"x": "Importance", "y": ""})
        impfig.update_layout(height=480)
        st.plotly_chart(impfig, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3 — LCA
# ---------------------------------------------------------------------------
with tab_lca:
    st.subheader("Latent Class Analysis on categorical-recoded customers")
    st.caption("Why LCA and not K-Means: most variables are categorical (Family, Education, "
               "account flags) and the continuous ones are binned into ordered bands, so a "
               "model designed for categorical indicators is appropriate. K-Means assumes "
               "Euclidean distance on continuous data and would mishandle these.")

    cset1, cset2 = st.columns([1, 2])
    with cset1:
        k = st.slider("Number of latent classes (k)", 2, 8, 3, 1)
    with cset2:
        st.markdown("**Model selection (BIC↓ / AIC↓ better, Rel. entropy↑ = cleaner separation)**")
        scan = lca_scan(path, 2, 8)
        st.dataframe(scan.set_index("k").style.format(
            {"BIC": "{:.0f}", "AIC": "{:.0f}", "Rel. entropy": "{:.3f}"}),
            use_container_width=True)

    with st.spinner(f"Fitting LCA with k={k}…"):
        labels, post, bic, aic, ent = fit_lca(path, k)

    seg = pd.DataFrame({"Class": labels, "Personal Loan": raw["Personal Loan"].values})
    rate = seg.groupby("Class")["Personal Loan"].agg(["mean", "size"]).rename(
        columns={"mean": "Loan rate", "size": "Customers"})
    rate["Lift vs base"] = rate["Loan rate"] / base_rate
    # Order classes by loan propensity for readability
    order = rate["Loan rate"].sort_values(ascending=False).index.tolist()

    m1, m2, m3 = st.columns(3)
    m1.metric("BIC", f"{bic:,.0f}")
    m2.metric("AIC", f"{aic:,.0f}")
    m3.metric("Rel. entropy", f"{ent:.3f}")

    st.subheader("Loan acceptance rate by latent class")
    bar = px.bar(rate.reset_index(), x="Class", y="Loan rate", text="Customers",
                 color="Loan rate", color_continuous_scale="Reds")
    bar.add_hline(y=base_rate, line_dash="dash",
                  annotation_text=f"Base rate {base_rate:.1%}")
    bar.update_layout(height=380)
    st.plotly_chart(bar, use_container_width=True)
    hi = order[0]
    st.success(f"**Class {hi}** has the highest propensity: "
               f"**{rate.loc[hi,'Loan rate']:.1%}** acceptance "
               f"({rate.loc[hi,'Lift vs base']:.1f}× the base rate). "
               "Non-acceptors inside high-propensity classes are prime look-alike prospects.")

    st.subheader("3-D segment map (PCA of the categorical profile)")
    comps, evr = pca_3d(path)
    plot_df = pd.DataFrame(comps, columns=["PC1", "PC2", "PC3"])
    plot_df["Class"] = labels.astype(str)
    plot_df["Loan"] = raw["Personal Loan"].map({0: "No", 1: "Accepted"}).values
    sym = st.checkbox("Mark actual loan-acceptors with ◆", value=True)
    scat = px.scatter_3d(
        plot_df, x="PC1", y="PC2", z="PC3", color="Class",
        symbol="Loan" if sym else None,
        symbol_map={"Accepted": "diamond", "No": "circle"} if sym else None,
        opacity=0.6, height=620)
    scat.update_traces(marker=dict(size=3))
    scat.update_layout(scene=dict(
        xaxis_title=f"PC1 ({evr[0]:.0%})",
        yaxis_title=f"PC2 ({evr[1]:.0%})",
        zaxis_title=f"PC3 ({evr[2]:.0%})"))
    st.plotly_chart(scat, use_container_width=True)
    st.caption(f"PCA is for visualisation only (first 3 PCs explain "
               f"{evr[:3].sum():.0%} of variance in the one-hot categorical space); "
               "the LCA itself uses all categories.")

    st.subheader("Descriptive profile per latent class")
    profile_src = model_df.copy()
    profile_src["Class"] = labels
    desc = profile_src.groupby("Class").agg(
        Customers=("Personal Loan", "size"),
        Loan_rate=("Personal Loan", "mean"),
        Avg_Income=("Income", "mean"),
        Avg_CCAvg=("CCAvg", "mean"),
        Avg_Family=("Family", "mean"),
        Avg_Education=("Education", "mean"),
        Has_CD=("CD Account", "mean"),
        Has_Securities=("Securities Account", "mean"),
        Online=("Online", "mean"),
    ).reindex(order)
    st.dataframe(desc.style.format({
        "Loan_rate": "{:.1%}", "Avg_Income": "{:.0f}", "Avg_CCAvg": "{:.2f}",
        "Avg_Family": "{:.1f}", "Avg_Education": "{:.2f}",
        "Has_CD": "{:.1%}", "Has_Securities": "{:.1%}", "Online": "{:.1%}"}),
        use_container_width=True)

    # stash for tab 4
    st.session_state["lca_labels"] = labels
    st.session_state["lca_rate"] = rate
    st.session_state["lca_k"] = k

# ---------------------------------------------------------------------------
# TAB 4 — MISSED LOOK-ALIKES
# ---------------------------------------------------------------------------
with tab_missed:
    st.subheader("Customers a classifier would miss — but who sit in high-propensity segments")
    st.caption("Logic: take the customers a chosen classifier predicts **No**, then keep those "
               "whose **latent class** has an above-base loan rate. These are statistical "
               "look-alikes of acceptors that the supervised model scores low — candidates for "
               "manual review or a softer-touch campaign.")

    if "lca_labels" not in st.session_state:
        st.warning("Open Tab 3 first to fit the latent classes.")
        st.stop()

    results, _ = train_classifiers(path)
    labels = st.session_state["lca_labels"]
    rate = st.session_state["lca_rate"]

    c1, c2 = st.columns(2)
    with c1:
        clf_name = st.selectbox("Classifier", list(results.keys()), index=1)
    with c2:
        thr = st.slider("Min. class loan-rate to count as 'high propensity'",
                        0.05, 0.50, max(0.15, float(base_rate * 1.5)), 0.01)

    full = model_df.copy()
    full.insert(0, "ID", raw["ID"].values)
    full["Actual_loan"] = raw["Personal Loan"].values
    full["LCA_class"] = labels
    full["Class_loan_rate"] = full["LCA_class"].map(rate["Loan rate"])
    full["Pred_loan"] = results[clf_name]["pred_all"]
    full["Pred_proba"] = results[clf_name]["proba_all"].round(3)

    missed = full[(full["Pred_loan"] == 0) & (full["Class_loan_rate"] >= thr)] \
        .sort_values(["Class_loan_rate", "Pred_proba"], ascending=[False, False])

    actual_fn = int(((full["Pred_loan"] == 0) & (full["Actual_loan"] == 1)).sum())
    fn_recovered = int(((missed["Actual_loan"] == 1)).sum())

    k1, k2, k3 = st.columns(3)
    k1.metric("Predicted-No look-alikes surfaced", f"{len(missed):,}")
    k2.metric(f"{clf_name} false negatives (total)", f"{actual_fn}")
    k3.metric("…of which appear in this list", f"{fn_recovered}",
              help="True acceptors the model missed that this segment view re-surfaces.")

    cols = ["ID", "Age", "Income", "Family", "Education", "CCAvg",
            "CD Account", "Securities Account", "Online", "CreditCard",
            "LCA_class", "Class_loan_rate", "Pred_proba", "Actual_loan"]
    st.dataframe(
        missed[cols].style.format({"Class_loan_rate": "{:.1%}", "CCAvg": "{:.1f}"}),
        use_container_width=True, height=460)

    st.download_button(
        "Download look-alike list (CSV)",
        missed[cols].to_csv(index=False).encode(),
        file_name="loan_lookalike_prospects.csv", mime="text/csv")
    st.caption("This is a prospecting aid, not a credit decision. The list mixes true acceptors "
               "the model missed with genuine non-acceptors who merely resemble them — both are "
               "worth a second look, but treat membership as a propensity signal, not a verdict.")
