"""
Script 02: Three-regime chemometric benchmark
=============================================
Runs the full benchmark on the rebuilt corpus:

  validation regimes   compound-level stratified CV, seed-grouped CV
                       (cyclic and class-stratified group assignment),
                       external holdout, out-of-scope probe
  label regimes        L0 as-submitted element heuristics
                       L1 CWC-calibrated
                       L2 CWC-calibrated and scaffold-filtered
  feature schemes      F1 physicochemical (8)
                       F2 physicochemical + elemental composition (20)
                       F3 F2 + CWC substructure fingerprints
  classifiers          DT, SVM (linear), LR, RF, XGB

and the diagnostics requested in review: variance-inflation factors, per-class
precision/recall with precision-recall and ROC curves under the conservative
regimes, probability calibration, permutation importance and SHAP values, and
a decomposition of the seed-grouped cross-validation variance.

Outputs are written as CSV/JSON under --outdir for the figure and table
scripts to consume; nothing is hard-coded into the manuscript.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cwc_schedules as cwc  # noqa: E402

warnings.filterwarnings("ignore")

SEEDS = [42, 7, 123, 256, 999]
N_FOLDS = 5
N_BOOT = 1000
RS = 42

PHYS = ["mw", "xlogp", "tpsa", "hbd", "hba", "rbc", "hac", "complexity"]
ELEM = ["has_P", "has_F", "has_S", "has_N", "has_Cl", "has_As",
        "has_PF", "has_PS", "has_SN", "P_ratio", "halogen_cnt", "S_cnt"]
CWC_ENTRY = [f for f in cwc.CWC_FEATURE_NAMES if f.startswith("cwc_")]
CWC_MOTIF = [f for f in cwc.CWC_FEATURE_NAMES if f.startswith("sca_")]

SCHEMES = {
    "F1": PHYS,
    "F2": PHYS + ELEM,
    "F3": PHYS + ELEM + CWC_ENTRY + CWC_MOTIF,
    # hierarchical ablation intermediates
    "F2+entry": PHYS + ELEM + CWC_ENTRY,
    "F2+motif": PHYS + ELEM + CWC_MOTIF,
}
CLASS_NAMES = ["CWA", "Simulant", "Pesticide"]


def make_clf(name: str, proba: bool = False):
    if name == "DT":
        return DecisionTreeClassifier(criterion="gini", max_depth=5,
                                      min_samples_leaf=4, random_state=RS)
    if name == "SVM":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", SVC(kernel="linear", C=1.0,
                                     decision_function_shape="ovr",
                                     probability=proba, random_state=RS))])
    if name == "LR":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(C=1.0, max_iter=5000,
                                                    solver="lbfgs",
                                                    random_state=RS))])
    if name == "RF":
        return RandomForestClassifier(n_estimators=200, criterion="gini",
                                      random_state=RS, n_jobs=-1)
    if name == "XGB":
        return XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                             eval_metric="mlogloss", random_state=RS,
                             verbosity=0, n_jobs=-1)
    raise ValueError(name)


CLFS = ["DT", "SVM", "LR", "RF", "XGB"]


def scores_matrix(clf, X) -> np.ndarray:
    """Class scores usable for ROC/PR, whichever the estimator provides."""
    if hasattr(clf, "predict_proba"):
        try:
            return clf.predict_proba(X)
        except (AttributeError, NotImplementedError):
            pass
    d = clf.decision_function(X)
    if d.ndim == 1:
        d = np.c_[-d, d]
    e = np.exp(d - d.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def boot_ci(vals, n_boot=N_BOOT, alpha=0.05):
    rng = np.random.default_rng(0)
    vals = np.asarray(vals, dtype=float)
    b = [rng.choice(vals, size=len(vals), replace=True).mean()
         for _ in range(n_boot)]
    return float(np.percentile(b, 100 * alpha / 2)), \
        float(np.percentile(b, 100 * (1 - alpha / 2)))


def macro_auc(y_true, S, n_cls=3):
    """Macro one-vs-rest ROC-AUC and average precision, NaN-safe."""
    ro, pr = [], []
    for c in range(n_cls):
        yb = (y_true == c).astype(int)
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        ro.append(roc_auc_score(yb, S[:, c]))
        pr.append(average_precision_score(yb, S[:, c]))
    return (float(np.mean(ro)) if ro else np.nan,
            float(np.mean(pr)) if pr else np.nan)


# ══════════════════════════════════════════════════════════════════════════
#  fold generators
# ══════════════════════════════════════════════════════════════════════════
def folds_compound(X, y, groups, seed):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    return list(skf.split(X, y))


def folds_seed_cyclic(X, y, groups, seed):
    """Original scheme: unique seed CIDs shuffled and assigned cyclically."""
    uniq = np.unique(groups)
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(uniq)
    fmap = {cid: i % N_FOLDS for i, cid in enumerate(shuffled)}
    fl = np.array([fmap[g] for g in groups])
    return [(np.where(fl != k)[0], np.where(fl == k)[0]) for k in range(N_FOLDS)]


def folds_seed_stratified(X, y, groups, seed):
    """Group-wise folds that additionally balance class composition."""
    sgk = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                               random_state=seed)
    return list(sgk.split(X, y, groups))


REGIMES = {
    "compound_cv": folds_compound,
    "seed_grouped_cv": folds_seed_cyclic,
    "seed_grouped_cv_stratified": folds_seed_stratified,
}


# ══════════════════════════════════════════════════════════════════════════
#  cross-validation driver
# ══════════════════════════════════════════════════════════════════════════
def run_cv(X, y, groups, clf_name, regime, collect_oof=False):
    gen = REGIMES[regime]
    accs, f1s, kappas, rocs, prs = [], [], [], [], []
    per_seed_acc, fold_records = [], []
    oof_y, oof_S = [], []
    for seed in SEEDS:
        seed_accs = []
        for fi, (tr, te) in enumerate(gen(X, y, groups, seed)):
            if len(te) == 0 or len(np.unique(y[tr])) < 2:
                continue
            clf = make_clf(clf_name)
            clf.fit(X[tr], y[tr])
            yp = clf.predict(X[te])
            S = scores_matrix(clf, X[te])
            a = accuracy_score(y[te], yp)
            accs.append(a)
            seed_accs.append(a)
            f1s.append(f1_score(y[te], yp, average="macro", zero_division=0))
            kappas.append(cohen_kappa_score(y[te], yp))
            r, p = macro_auc(y[te], S)
            rocs.append(r)
            prs.append(p)
            fold_records.append({
                "seed": seed, "fold": fi, "n_test": int(len(te)),
                "acc": a,
                "n_cwa": int((y[te] == 0).sum()),
                "n_sim": int((y[te] == 1).sum()),
                "n_pest": int((y[te] == 2).sum()),
                "majority_frac": float(np.bincount(y[te], minlength=3).max()
                                       / len(te)),
            })
            if collect_oof and seed == SEEDS[0]:
                oof_y.append(y[te])
                oof_S.append(S)
        if seed_accs:
            per_seed_acc.append(float(np.mean(seed_accs)))

    lo, hi = boot_ci(accs)
    out = {
        "acc_mean": 100 * float(np.mean(accs)),
        "acc_sd": 100 * float(np.std(accs, ddof=0)),
        "acc_ci_lo": 100 * lo, "acc_ci_hi": 100 * hi,
        "macro_f1": float(np.mean(f1s)),
        "kappa": float(np.mean(kappas)),
        "roc_auc_macro": float(np.nanmean(rocs)),
        "pr_auc_macro": float(np.nanmean(prs)),
        "n_evals": len(accs),
        "per_seed_acc": per_seed_acc,
        "fold_records": fold_records,
    }
    if collect_oof and oof_y:
        out["oof_y"] = np.concatenate(oof_y)
        out["oof_S"] = np.vstack(oof_S)
    return out


def eval_external(Xtr, ytr, Xte, yte, clf_name):
    clf = make_clf(clf_name)
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xte)
    S = scores_matrix(clf, Xte)
    roc, pr = macro_auc(yte, S)
    maj = int(np.bincount(ytr, minlength=3).argmax())
    p, r, f, sup = precision_recall_fscore_support(
        yte, yp, labels=[0, 1, 2], zero_division=0)
    return {
        "acc": 100 * accuracy_score(yte, yp),
        "n_correct": int((yp == yte).sum()), "n": int(len(yte)),
        "balanced_acc": float(balanced_accuracy_score(yte, yp)),
        "mcc": float(matthews_corrcoef(yte, yp)),
        "macro_f1": float(f1_score(yte, yp, average="macro", zero_division=0)),
        "roc_auc_macro": roc, "pr_auc_macro": pr,
        "majority_baseline": 100 * float((yte == maj).mean()),
        "per_class_precision": p.tolist(), "per_class_recall": r.tolist(),
        "per_class_f1": f.tolist(), "per_class_support": sup.tolist(),
        "confusion": confusion_matrix(yte, yp, labels=[0, 1, 2]).tolist(),
        "y_pred": yp.tolist(), "scores": S.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════
#  diagnostics
# ══════════════════════════════════════════════════════════════════════════
def compute_vif(df, cols):
    """Variance inflation factors via the R^2 of each feature on the rest."""
    from sklearn.linear_model import LinearRegression
    Z = df[cols].astype(float).values
    keep = [i for i in range(Z.shape[1]) if np.std(Z[:, i]) > 0]
    rows = []
    for j, c in enumerate(cols):
        if j not in keep:
            rows.append({"feature": c, "vif": np.nan, "note": "zero variance"})
            continue
        others = [i for i in keep if i != j]
        r2 = LinearRegression().fit(Z[:, others], Z[:, j]).score(
            Z[:, others], Z[:, j])
        r2 = min(max(r2, 0.0), 1 - 1e-12)
        rows.append({"feature": c, "vif": float(1.0 / (1.0 - r2)), "note": ""})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def greedy_vif_prune(df, cols, thresh=10.0):
    """Iteratively drop the highest-VIF feature until all VIF < threshold."""
    cur = list(cols)
    dropped = []
    while len(cur) > 2:
        v = compute_vif(df, cur).dropna(subset=["vif"])
        if v.empty or v.iloc[0]["vif"] < thresh:
            break
        worst = v.iloc[0]["feature"]
        dropped.append({"feature": worst, "vif": float(v.iloc[0]["vif"])})
        cur.remove(worst)
    return cur, dropped


def calibration_stats(y_true, S, n_bins=10):
    """Per-class reliability bins, Brier score and expected calibration error."""
    out = {"bins": [], "brier": {}, "ece": {}}
    for c in range(S.shape[1]):
        yb = (y_true == c).astype(int)
        p = S[:, c]
        out["brier"][CLASS_NAMES[c]] = float(brier_score_loss(yb, p))
        edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for k in range(n_bins):
            m = (p >= edges[k]) & (p < edges[k + 1] if k < n_bins - 1
                                   else p <= edges[k + 1])
            if m.sum() == 0:
                continue
            conf, acc = float(p[m].mean()), float(yb[m].mean())
            ece += (m.sum() / len(p)) * abs(conf - acc)
            out["bins"].append({"class": CLASS_NAMES[c], "bin": k,
                                "n": int(m.sum()), "confidence": conf,
                                "observed": acc})
        out["ece"][CLASS_NAMES[c]] = float(ece)
    return out


def curve_points(y_true, S):
    out = {"pr": {}, "roc": {}}
    for c in range(S.shape[1]):
        yb = (y_true == c).astype(int)
        if yb.sum() == 0:
            continue
        pre, rec, _ = precision_recall_curve(yb, S[:, c])
        fpr, tpr, _ = roc_curve(yb, S[:, c])
        step = max(1, len(rec) // 400)
        out["pr"][CLASS_NAMES[c]] = {
            "recall": rec[::step].tolist(), "precision": pre[::step].tolist(),
            "ap": float(average_precision_score(yb, S[:, c])),
            "baseline": float(yb.mean())}
        step = max(1, len(fpr) // 400)
        out["roc"][CLASS_NAMES[c]] = {
            "fpr": fpr[::step].tolist(), "tpr": tpr[::step].tolist(),
            "auc": float(roc_auc_score(yb, S[:, c]))}
    return out


def variance_decomposition(fold_records):
    """Split seed-grouped CV variance into between-seed and within-seed parts.

    Reported to explain the large standard deviations observed under
    group-wise cross-validation: most of the spread arises from how the
    class composition of a held-out fold varies, not from model instability.
    """
    df = pd.DataFrame(fold_records)
    grand = df["acc"].mean()
    seed_means = df.groupby("seed")["acc"].mean()
    between = float(((seed_means - grand) ** 2).mean())
    within = float(df.groupby("seed")["acc"].var(ddof=0).mean())
    corr = df[["acc", "majority_frac"]].corr().iloc[0, 1]
    return {
        "grand_mean_acc": float(grand),
        "sd_all_folds": float(df["acc"].std(ddof=0)),
        "sd_seed_means": float(seed_means.std(ddof=0)),
        "var_between_seeds": between,
        "var_within_seed_between_folds": within,
        "frac_variance_within_seed": (
            within / (within + between) if (within + between) > 0 else np.nan),
        "corr_acc_vs_fold_majority_fraction": float(corr),
        "fold_majority_frac_min": float(df["majority_frac"].min()),
        "fold_majority_frac_max": float(df["majority_frac"].max()),
        "fold_n_test_min": int(df["n_test"].min()),
        "fold_n_test_max": int(df["n_test"].max()),
    }


# ══════════════════════════════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/build/corpus.csv")
    ap.add_argument("--holdout", default="data/build/holdout.csv")
    ap.add_argument("--probe", default="data/build/probe.csv")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    corpus = pd.read_csv(a.corpus)
    hold = pd.read_csv(a.holdout)
    probe = pd.read_csv(a.probe)

    label_sets = {
        "L0": (corpus, "label_L0"),
        "L1": (corpus, "label_L1"),
        "L2": (corpus[corpus.scaffold_ok == 1].reset_index(drop=True),
               "label_L2"),
    }
    y_hold = hold["inherited_label"].values

    results, holdout_rows, probe_rows = [], [], []
    extras: dict = {}

    # ── main sweep ────────────────────────────────────────────────────────
    sweep = [("L0", "F2"), ("L1", "F2"), ("L2", "F2"),
             ("L2", "F1"), ("L2", "F3"),
             ("L2", "F2+entry"), ("L2", "F2+motif"),
             ("L0", "F3"), ("L1", "F3")]
    for lab, sch in sweep:
        df, ycol = label_sets[lab]
        cols = SCHEMES[sch]
        X = df[cols].astype(float).values
        y = df[ycol].astype(int).values
        g = df["seed_cid"].values
        Xh = hold[cols].astype(float).values
        Xp = probe[cols].astype(float).values
        print(f"\n== labels {lab}  features {sch} "
              f"({len(cols)} cols, n={len(df)}) ==", flush=True)
        for cn in CLFS:
            row = {"label_regime": lab, "feature_scheme": sch,
                   "n_features": len(cols), "n_train": int(len(df)),
                   "classifier": cn}
            for reg in REGIMES:
                want_oof = (lab == "L2" and sch == "F3"
                            and reg == "seed_grouped_cv")
                r = run_cv(X, y, g, cn, reg, collect_oof=want_oof)
                row[f"{reg}_acc"] = r["acc_mean"]
                row[f"{reg}_sd"] = r["acc_sd"]
                row[f"{reg}_ci_lo"] = r["acc_ci_lo"]
                row[f"{reg}_ci_hi"] = r["acc_ci_hi"]
                row[f"{reg}_f1"] = r["macro_f1"]
                row[f"{reg}_kappa"] = r["kappa"]
                row[f"{reg}_roc"] = r["roc_auc_macro"]
                row[f"{reg}_pr"] = r["pr_auc_macro"]
                if reg == "seed_grouped_cv":
                    extras.setdefault("per_seed_acc", {})[
                        f"{lab}|{sch}|{cn}"] = r["per_seed_acc"]
                    if lab == "L2" and sch == "F3":
                        extras.setdefault("variance", {})[cn] = \
                            variance_decomposition(r["fold_records"])
                        extras.setdefault("fold_records", {})[cn] = \
                            r["fold_records"]
                if want_oof and "oof_y" in r:
                    extras.setdefault("curves_seedgrouped", {})[cn] = \
                        curve_points(r["oof_y"], r["oof_S"])
                    extras.setdefault("calibration_seedgrouped", {})[cn] = \
                        calibration_stats(r["oof_y"], r["oof_S"])
                    extras.setdefault("confusion_seedgrouped", {})[cn] = \
                        confusion_matrix(r["oof_y"],
                                         r["oof_S"].argmax(1),
                                         labels=[0, 1, 2],
                                         normalize="true").tolist()
            # external holdout
            e = eval_external(X, y, Xh, y_hold, cn)
            for k, v in e.items():
                if k in ("y_pred", "scores"):
                    continue
                row[f"holdout_{k}"] = v
            results.append(row)
            print(f"   {cn:4s} cmpd {row['compound_cv_acc']:5.1f}%  "
                  f"seedgrp {row['seed_grouped_cv_acc']:5.1f}%  "
                  f"seedgrp-str {row['seed_grouped_cv_stratified_acc']:5.1f}%  "
                  f"holdout {row['holdout_acc']:5.1f}%", flush=True)

            if lab == "L2" and sch == "F3":
                S = np.array(e["scores"])
                extras.setdefault("curves_holdout", {})[cn] = \
                    curve_points(y_hold, S)
                extras.setdefault("calibration_holdout", {})[cn] = \
                    calibration_stats(y_hold, S)
                for i, cid in enumerate(hold["cid"].values):
                    holdout_rows.append({
                        "classifier": cn, "cid": int(cid),
                        "name": hold["name"].iloc[i],
                        "subclass": hold["subclass"].iloc[i],
                        "formula": hold["formula"].iloc[i],
                        "smiles": hold["smiles"].iloc[i],
                        "cwc_schedule": int(hold["cwc_schedule"].iloc[i]),
                        "cwc_entries": hold["cwc_entries"].iloc[i],
                        "true": int(y_hold[i]),
                        "pred": int(e["y_pred"][i]),
                        "correct": bool(e["y_pred"][i] == y_hold[i]),
                        "p_cwa": float(S[i, 0]), "p_sim": float(S[i, 1]),
                        "p_pest": float(S[i, 2]),
                    })
                clf = make_clf(cn)
                clf.fit(X, y)
                Sp = scores_matrix(clf, Xp)
                pp = clf.predict(Xp)
                for i, cid in enumerate(probe["cid"].values):
                    probe_rows.append({
                        "classifier": cn, "cid": int(cid),
                        "name": probe["name"].iloc[i],
                        "formula": probe["formula"].iloc[i],
                        "cwc_schedule": int(probe["cwc_schedule"].iloc[i]),
                        "pred": int(pp[i]),
                        "pred_class": CLASS_NAMES[int(pp[i])],
                        "p_cwa": float(Sp[i, 0]), "p_sim": float(Sp[i, 1]),
                        "p_pest": float(Sp[i, 2]),
                        "max_prob": float(Sp[i].max()),
                    })

    res = pd.DataFrame(results)
    res.to_csv(out / "benchmark.csv", index=False)
    pd.DataFrame(holdout_rows).to_csv(out / "holdout_predictions.csv",
                                      index=False)
    pd.DataFrame(probe_rows).to_csv(out / "probe_predictions.csv", index=False)

    # ── VIF and collinearity pruning ──────────────────────────────────────
    print("\n== variance inflation factors ==", flush=True)
    df2, _ = label_sets["L2"]
    vif_f2 = compute_vif(df2, SCHEMES["F2"])
    vif_f3 = compute_vif(df2, SCHEMES["F3"])
    vif_f2.assign(scheme="F2").to_csv(out / "vif_F2.csv", index=False)
    vif_f3.assign(scheme="F3").to_csv(out / "vif_F3.csv", index=False)
    kept, dropped = greedy_vif_prune(df2, SCHEMES["F3"], thresh=10.0)
    extras["vif_pruned_kept"] = kept
    extras["vif_pruned_dropped"] = dropped
    print(vif_f3.head(12).to_string(index=False))
    print(f"   pruning to VIF<10 drops {len(dropped)} of "
          f"{len(SCHEMES['F3'])} features", flush=True)

    # benchmark on the VIF-pruned feature set and with stronger L2 penalty
    X = df2[kept].astype(float).values
    y = df2["label_L2"].astype(int).values
    g = df2["seed_cid"].values
    Xh = hold[kept].astype(float).values
    pruned_rows = []
    for cn in CLFS:
        r = run_cv(X, y, g, cn, "seed_grouped_cv")
        e = eval_external(X, y, Xh, y_hold, cn)
        pruned_rows.append({"classifier": cn, "n_features": len(kept),
                            "seed_grouped_acc": r["acc_mean"],
                            "seed_grouped_sd": r["acc_sd"],
                            "holdout_acc": e["acc"],
                            "holdout_mcc": e["mcc"]})
    pd.DataFrame(pruned_rows).to_csv(out / "vif_pruned_benchmark.csv",
                                     index=False)

    Xfull = df2[SCHEMES["F3"]].astype(float).values
    Xhf = hold[SCHEMES["F3"]].astype(float).values
    reg_rows = []
    for C in [0.01, 0.1, 1.0, 10.0]:
        clf = Pipeline([("sc", StandardScaler()),
                        ("clf", LogisticRegression(C=C, penalty="l2",
                                                   max_iter=5000,
                                                   random_state=RS))])
        accs = []
        for seed in SEEDS:
            for tr, te in folds_seed_cyclic(Xfull, y, g, seed):
                clf.fit(Xfull[tr], y[tr])
                accs.append(accuracy_score(y[te], clf.predict(Xfull[te])))
        clf.fit(Xfull, y)
        reg_rows.append({"penalty": "l2", "C": C,
                         "seed_grouped_acc": 100 * float(np.mean(accs)),
                         "holdout_acc": 100 * accuracy_score(
                             y_hold, clf.predict(Xhf))})
    pd.DataFrame(reg_rows).to_csv(out / "regularisation_sweep.csv", index=False)

    # ── feature importance: Gini, permutation, SHAP ────────────────────────
    print("\n== feature importance ==", flush=True)
    imp_rows = []
    for cn in ["DT", "RF", "XGB"]:
        clf = make_clf(cn)
        clf.fit(Xfull, y)
        gini = getattr(clf, "feature_importances_", None)
        perm = permutation_importance(clf, Xfull, y, n_repeats=20,
                                      random_state=RS, n_jobs=-1,
                                      scoring="accuracy")
        for i, f in enumerate(SCHEMES["F3"]):
            imp_rows.append({
                "classifier": cn, "feature": f,
                "block": ("phys" if f in PHYS else
                          "elem" if f in ELEM else
                          "cwc_entry" if f in CWC_ENTRY else "cwc_motif"),
                "gini": float(gini[i]) if gini is not None else np.nan,
                "perm_mean": float(perm.importances_mean[i]),
                "perm_sd": float(perm.importances_std[i]),
            })
    imp = pd.DataFrame(imp_rows)

    try:
        import shap
        clf = make_clf("RF")
        clf.fit(Xfull, y)
        expl = shap.TreeExplainer(clf)
        idx = np.random.default_rng(RS).choice(
            len(Xfull), size=min(600, len(Xfull)), replace=False)
        sv = expl.shap_values(Xfull[idx], check_additivity=False)
        arr = np.array(sv)
        if arr.ndim == 3 and arr.shape[0] == len(idx):      # (n, f, c)
            mean_abs = np.abs(arr).mean(axis=(0, 2))
        elif arr.ndim == 3:                                  # (c, n, f)
            mean_abs = np.abs(arr).mean(axis=(0, 1))
        else:
            mean_abs = np.abs(arr).mean(axis=0)
        shap_map = {f: float(mean_abs[i])
                    for i, f in enumerate(SCHEMES["F3"])}
        imp["shap_mean_abs"] = imp.apply(
            lambda r: shap_map.get(r["feature"], np.nan)
            if r["classifier"] == "RF" else np.nan, axis=1)
        extras["shap_ok"] = True
    except Exception as exc:                                 # pragma: no cover
        print("   SHAP unavailable:", exc)
        imp["shap_mean_abs"] = np.nan
        extras["shap_ok"] = False
    imp.to_csv(out / "feature_importance.csv", index=False)

    # ── pairwise classifier tests on seed-level means ─────────────────────
    tests = []
    for reg in ["compound_cv", "seed_grouped_cv"]:
        for i, ci in enumerate(CLFS):
            for cj in CLFS[i + 1:]:
                ai = extras["per_seed_acc"].get(f"L2|F3|{ci}")
                aj = extras["per_seed_acc"].get(f"L2|F3|{cj}")
                if not ai or not aj:
                    continue
                key = f"L2|F3|"
                row_i = res[(res.label_regime == "L2") & (res.feature_scheme == "F3")
                            & (res.classifier == ci)][f"{reg}_acc"].values
                row_j = res[(res.label_regime == "L2") & (res.feature_scheme == "F3")
                            & (res.classifier == cj)][f"{reg}_acc"].values
                if reg == "seed_grouped_cv":
                    x, yy = np.array(ai), np.array(aj)
                    sh = stats.shapiro(x - yy).pvalue if len(x) >= 3 else np.nan
                    if np.isnan(sh) or sh > 0.05:
                        t, p = stats.ttest_rel(x, yy)
                        test = "paired t"
                    else:
                        t, p = stats.mannwhitneyu(x, yy)
                        test = "Mann-Whitney U"
                    d = ((x - yy).mean() / (x - yy).std(ddof=1)
                         if (x - yy).std(ddof=1) > 0 else np.nan)
                    tests.append({"regime": reg, "a": ci, "b": cj,
                                  "mean_a": 100 * float(x.mean()),
                                  "mean_b": 100 * float(yy.mean()),
                                  "test": test, "stat": float(t),
                                  "p": float(p), "cohens_d": float(d),
                                  "shapiro_p": float(sh) if sh == sh else np.nan})
                else:
                    tests.append({"regime": reg, "a": ci, "b": cj,
                                  "mean_a": float(row_i[0]) if len(row_i) else np.nan,
                                  "mean_b": float(row_j[0]) if len(row_j) else np.nan,
                                  "test": "", "stat": np.nan, "p": np.nan,
                                  "cohens_d": np.nan, "shapiro_p": np.nan})
    pd.DataFrame(tests).to_csv(out / "pairwise_tests.csv", index=False)

    # ── binomial tests on the external holdout ────────────────────────────
    bino = []
    for cn in CLFS:
        r = res[(res.label_regime == "L2") & (res.feature_scheme == "F3")
                & (res.classifier == cn)].iloc[0]
        k, n = int(r["holdout_n_correct"]), int(r["holdout_n"])
        maj = r["holdout_majority_baseline"] / 100
        bino.append({
            "classifier": cn, "n_correct": k, "n": n,
            "acc": 100 * k / n,
            "ci_lo": 100 * stats.beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0,
            "ci_hi": 100 * stats.beta.ppf(0.975, k + 1, n - k) if k < n else 100.0,
            "p_vs_uniform": float(stats.binomtest(k, n, 1 / 3,
                                                  alternative="greater").pvalue),
            "p_vs_majority": float(stats.binomtest(k, n, maj,
                                                   alternative="greater").pvalue),
        })
    pd.DataFrame(bino).to_csv(out / "holdout_binomial.csv", index=False)

    with open(out / "extras.json", "w") as fh:
        json.dump(extras, fh, indent=2, default=float)

    print("\n  wrote results to", out.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
