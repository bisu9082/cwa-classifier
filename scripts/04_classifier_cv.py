"""
Script 04: Classifier Training, Cross-Validation & Evaluation
=============================================================
Benchmarks five classifiers (DT, SVM, LR, RF, XGB) under two
cross-validation regimes and a pilot external holdout.

CV Regimes
----------
1. Compound-level CV  : standard stratified 5-fold × 5 random seeds
                        (susceptible to within-seed-family leakage)
2. Seed-grouped CV    : compounds from the same seed CID are co-assigned
                        to the same fold (eliminates within-cluster leakage)

External holdout      : 20 OPCW-grounded / EPA-registered compounds held
                        completely outside the training corpus.

Input:  data/expanded_dataset.csv
Output: results/cv_results.csv
        results/holdout_results.csv

Usage:
    python scripts/04_classifier_cv.py \
        --input   data/expanded_dataset.csv \
        --holdout data/external_holdout.csv \
        --outdir  results

Reference: Code Listing S4 in Supplementary Information.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score,
                              cohen_kappa_score, balanced_accuracy_score,
                              matthews_corrcoef, roc_auc_score)

# ── Configuration ──────────────────────────────────────────────────────────
SEEDS    = [42, 7, 123, 256, 999]
N_FOLDS  = 5
N_BOOT   = 1_000     # bootstrap replicates for CIs

PHYS_FEATURES = ['mw', 'xlogp', 'tpsa', 'hbd', 'hba', 'rbc', 'hac', 'complexity']
ELEM_FEATURES = ['has_P', 'has_F', 'has_S', 'has_N', 'has_Cl', 'has_As',
                 'has_PF', 'has_PS', 'has_SN', 'P_ratio', 'halogen_cnt', 'S_cnt']
ALL_FEATURES  = PHYS_FEATURES + ELEM_FEATURES   # 20-dimensional vector

CLASS_NAMES = ['CWA', 'Simulant', 'Pesticide']


def make_classifiers() -> dict:
    """Instantiate five classifier branches as described in the paper."""
    return {
        'DT':  DecisionTreeClassifier(
                   criterion='gini', max_depth=5, min_samples_leaf=4),
        'SVM': Pipeline([
                   ('scaler', StandardScaler()),
                   ('clf',    SVC(kernel='linear', C=1.0,
                                  decision_function_shape='ovr',
                                  probability=True))]),
        'LR':  Pipeline([
                   ('scaler', StandardScaler()),
                   ('clf',    LogisticRegression(
                                  C=1.0, max_iter=1_000,
                                  solver='lbfgs', multi_class='auto'))]),
        'RF':  RandomForestClassifier(
                   n_estimators=200, random_state=42, n_jobs=-1),
        'XGB': XGBClassifier(
                   n_estimators=200, max_depth=5, learning_rate=0.1,
                   eval_metric='mlogloss', random_state=42, verbosity=0),
    }


def bootstrap_ci(scores: list, n_boot: int = 1_000,
                 alpha: float = 0.05) -> tuple:
    """Compute bootstrap percentile confidence interval.

    Returns (lower, upper) at (alpha/2, 1-alpha/2) percentiles.
    """
    rng     = np.random.default_rng(0)
    boots   = [np.mean(rng.choice(scores, size=len(scores), replace=True))
               for _ in range(n_boot)]
    lo = np.percentile(boots, 100 * alpha / 2)
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return lo, hi


def run_compound_cv(X: np.ndarray, y: np.ndarray,
                    classifiers: dict) -> pd.DataFrame:
    """Compound-level stratified 5-fold CV (5 seeds × 5 folds = 25 evals)."""
    rows = []
    for name, clf in classifiers.items():
        all_accs, all_f1s, all_kappas = [], [], []
        for seed in SEEDS:
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                  random_state=seed)
            for tr_idx, te_idx in skf.split(X, y):
                clf_copy = make_classifiers()[name]   # fresh instance each fold
                clf_copy.fit(X[tr_idx], y[tr_idx])
                y_pred = clf_copy.predict(X[te_idx])
                all_accs.append(accuracy_score(y[te_idx], y_pred))
                all_f1s.append(f1_score(y[te_idx], y_pred, average='macro',
                                        zero_division=0))
                all_kappas.append(cohen_kappa_score(y[te_idx], y_pred))

        ci_lo, ci_hi = bootstrap_ci(all_accs, N_BOOT)
        rows.append({
            'classifier': name,
            'regime':     'compound_cv',
            'acc_mean':   np.mean(all_accs) * 100,
            'acc_sd':     np.std(all_accs)  * 100,
            'acc_ci_lo':  ci_lo * 100,
            'acc_ci_hi':  ci_hi * 100,
            'macro_f1':   np.mean(all_f1s),
            'kappa':      np.mean(all_kappas),
        })
        print(f'  {name:4s} compound-CV  Acc={np.mean(all_accs)*100:.1f}% '
              f'(95% CI {ci_lo*100:.1f}–{ci_hi*100:.1f}%)')
    return pd.DataFrame(rows)


def run_seed_grouped_cv(X: np.ndarray, y: np.ndarray,
                        seed_cids: np.ndarray,
                        classifiers: dict) -> pd.DataFrame:
    """Seed-grouped CV: compounds from the same seed CID share a fold."""
    unique_seeds = np.unique(seed_cids)
    rows = []
    for name, clf in classifiers.items():
        all_accs, all_f1s = [], []
        for seed in SEEDS:
            rng      = np.random.RandomState(seed)
            shuffled = rng.permutation(unique_seeds)
            fold_map = {cid: i % N_FOLDS
                        for i, cid in enumerate(shuffled)}
            fold_labels = np.array([fold_map[s] for s in seed_cids])

            for fold in range(N_FOLDS):
                tr_mask = fold_labels != fold
                te_mask = fold_labels == fold
                if te_mask.sum() == 0:
                    continue
                clf_copy = make_classifiers()[name]
                clf_copy.fit(X[tr_mask], y[tr_mask])
                y_pred = clf_copy.predict(X[te_mask])
                all_accs.append(accuracy_score(y[te_mask], y_pred))
                all_f1s.append(f1_score(y[te_mask], y_pred,
                                        average='macro', zero_division=0))

        ci_lo, ci_hi = bootstrap_ci(all_accs, N_BOOT)
        rows.append({
            'classifier': name,
            'regime':     'seed_grouped_cv',
            'acc_mean':   np.mean(all_accs) * 100,
            'acc_sd':     np.std(all_accs)  * 100,
            'acc_ci_lo':  ci_lo * 100,
            'acc_ci_hi':  ci_hi * 100,
            'macro_f1':   np.mean(all_f1s),
        })
        print(f'  {name:4s} seed-grouped  Acc={np.mean(all_accs)*100:.1f}% '
              f'(95% CI {ci_lo*100:.1f}–{ci_hi*100:.1f}%)')
    return pd.DataFrame(rows)


def run_external_holdout(X_train: np.ndarray, y_train: np.ndarray,
                         X_hold: np.ndarray,  y_hold: np.ndarray,
                         classifiers: dict) -> pd.DataFrame:
    """Train on full expanded dataset, evaluate on external holdout."""
    rows = []
    majority_class = np.bincount(y_train).argmax()
    baseline_acc   = (y_hold == majority_class).mean() * 100

    for name, clf in classifiers.items():
        clf_copy = make_classifiers()[name]
        clf_copy.fit(X_train, y_train)
        y_pred = clf_copy.predict(X_hold)
        n_correct = (y_pred == y_hold).sum()

        acc  = accuracy_score(y_hold, y_pred) * 100
        bacc = balanced_accuracy_score(y_hold, y_pred)
        mcc  = matthews_corrcoef(y_hold, y_pred)
        f1   = f1_score(y_hold, y_pred, average='macro', zero_division=0)

        rows.append({
            'classifier':   name,
            'acc':          acc,
            'n_correct':    int(n_correct),
            'balanced_acc': bacc,
            'mcc':          mcc,
            'macro_f1':     f1,
            'majority_baseline_acc': baseline_acc,
        })
        above = '> baseline' if acc > baseline_acc else '< baseline'
        print(f'  {name:4s} holdout  Acc={acc:.1f}%  MCC={mcc:+.3f}  [{above}]')

    print(f'  Majority-class baseline: {baseline_acc:.1f}%')
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description='Five-classifier benchmark with three validation regimes.')
    parser.add_argument('--input',   default='data/expanded_dataset.csv')
    parser.add_argument('--holdout', default='data/external_holdout.csv',
                        help='External holdout CSV (optional; skip if absent)')
    parser.add_argument('--outdir',  default='results',
                        help='Output directory for result CSVs')
    args = parser.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────
    df = pd.read_csv(args.input)
    print(f'Loaded {len(df)} compounds | class distribution:')
    print(df.groupby('class')['cid'].count().to_string(), '\n')

    X         = df[ALL_FEATURES].values.astype(float)
    y         = df['class_id'].values.astype(int)
    seed_cids = df['seed_cid'].values.astype(int)

    classifiers = make_classifiers()

    # ── Compound-level CV ──────────────────────────────────────────────
    print('=== Compound-level CV ===')
    cv_compound = run_compound_cv(X, y, classifiers)

    # ── Seed-grouped CV ────────────────────────────────────────────────
    print('\n=== Seed-grouped CV ===')
    cv_seed = run_seed_grouped_cv(X, y, seed_cids, classifiers)

    # Combine and save
    cv_all = pd.concat([cv_compound, cv_seed], ignore_index=True)
    cv_path = os.path.join(args.outdir, 'cv_results.csv')
    cv_all.to_csv(cv_path, index=False)
    print(f'\nCV results saved to {cv_path}')

    # Leakage gap summary
    print('\n=== Leakage Gap (Compound-level - Seed-grouped) ===')
    for name in classifiers:
        acc_comp = cv_compound.loc[cv_compound.classifier == name, 'acc_mean'].values[0]
        acc_seed = cv_seed.loc[cv_seed.classifier == name,     'acc_mean'].values[0]
        print(f'  {name:4s}: {acc_comp:.1f}% - {acc_seed:.1f}% = {acc_comp-acc_seed:.1f} pp')

    # ── External holdout ───────────────────────────────────────────────
    holdout_path = args.holdout
    if os.path.exists(holdout_path):
        hold_df = pd.read_csv(holdout_path)
        X_hold  = hold_df[ALL_FEATURES].values.astype(float)
        y_hold  = hold_df['class_id'].values.astype(int)
        print(f'\n=== External Holdout (n={len(hold_df)}) ===')
        holdout_results = run_external_holdout(X, y, X_hold, y_hold, classifiers)
        h_path = os.path.join(args.outdir, 'holdout_results.csv')
        holdout_results.to_csv(h_path, index=False)
        print(f'Holdout results saved to {h_path}')
    else:
        print(f'\n[INFO] Holdout file not found at {holdout_path}, skipping.')

    print('\nDone.')


if __name__ == '__main__':
    main()
