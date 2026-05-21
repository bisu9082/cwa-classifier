"""
Script 02: Elemental Composition Feature Extraction
====================================================
Parses molecular formula strings to extract 12 binary/count features
that encode chemically meaningful CWA-diagnostic structural motifs.

Feature set (canonical order):
  has_P, has_F, has_S, has_N, has_Cl, has_As,
  has_PF, has_PS, has_SN, P_ratio, halogen_cnt, S_cnt

Input:  any CSV with a 'formula' column
Output: same CSV + 12 new feature columns

Usage:
    python scripts/02_elemental_features.py \
        --input  data/seed_properties.csv \
        --output data/seed_features.csv

Reference: Code Listing S2 in Supplementary Information.
"""

import argparse
import re
from collections import defaultdict

import pandas as pd

# ── Feature names (canonical order) ───────────────────────────────────────
ELEM_FEAT_NAMES = [
    'has_P', 'has_F', 'has_S', 'has_N', 'has_Cl', 'has_As',
    'has_PF', 'has_PS', 'has_SN',
    'P_ratio', 'halogen_cnt', 'S_cnt',
]


def parse_formula(formula: str) -> dict:
    """Parse a molecular formula string into an element-count dictionary.

    Examples
    --------
    >>> parse_formula('C4H10FO2P')
    {'C': 4, 'H': 10, 'F': 1, 'O': 2, 'P': 1}
    >>> parse_formula('C2H2AsCl3')
    {'C': 2, 'H': 2, 'As': 1, 'Cl': 3}
    """
    if not formula or not isinstance(formula, str):
        return {}
    pattern = r'([A-Z][a-z]?)(\d*)'
    counts = defaultdict(int)
    for elem, cnt in re.findall(pattern, formula):
        counts[elem] += int(cnt) if cnt else 1
    return dict(counts)


def formula_to_elem_features(formula: str) -> list:
    """Extract 12 elemental composition features from a formula string.

    Returns
    -------
    List of 12 floats in canonical order (see ELEM_FEAT_NAMES).
    """
    e = parse_formula(formula)
    total_heavy = sum(v for k, v in e.items() if k != 'H')
    if total_heavy == 0:
        total_heavy = 1   # avoid division by zero for trivial molecules

    has_P  = int(e.get('P',  0) > 0)
    has_F  = int(e.get('F',  0) > 0)
    has_S  = int(e.get('S',  0) > 0)
    has_N  = int(e.get('N',  0) > 0)
    has_Cl = int(e.get('Cl', 0) > 0)
    has_As = int(e.get('As', 0) > 0)

    has_PF = int(has_P and has_F)    # G-agent motif
    has_PS = int(has_P and has_S)    # V-agent motif
    has_SN = int(has_S and has_N)

    P_ratio     = e.get('P', 0) / total_heavy
    halogen_cnt = (e.get('F', 0) + e.get('Cl', 0) +
                   e.get('Br', 0) + e.get('I', 0))
    S_cnt       = e.get('S', 0)

    return [has_P, has_F, has_S, has_N, has_Cl, has_As,
            has_PF, has_PS, has_SN,
            P_ratio, halogen_cnt, S_cnt]


def add_elemental_features(df: pd.DataFrame,
                            formula_col: str = 'formula') -> pd.DataFrame:
    """Append 12 elemental features to a DataFrame.

    Parameters
    ----------
    df          : input DataFrame with a formula column
    formula_col : name of the formula column (default 'formula')

    Returns
    -------
    DataFrame with 12 new feature columns appended.
    """
    feat_matrix = df[formula_col].apply(formula_to_elem_features).tolist()
    feat_df = pd.DataFrame(feat_matrix, columns=ELEM_FEAT_NAMES,
                           index=df.index)
    return pd.concat([df, feat_df], axis=1)


def main():
    parser = argparse.ArgumentParser(
        description='Add 12 elemental composition features from molecular formula.')
    parser.add_argument('--input',  default='data/seed_properties.csv',
                        help='Input CSV with formula column')
    parser.add_argument('--output', default='data/seed_features.csv',
                        help='Output CSV with 12 elemental features appended')
    parser.add_argument('--formula-col', default='formula',
                        help='Name of the formula column (default: formula)')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f'Loaded {len(df)} rows from {args.input}')

    df_out = add_elemental_features(df, formula_col=args.formula_col)

    n_feat = df_out[ELEM_FEAT_NAMES].notna().all(axis=1).sum()
    print(f'Features extracted for {n_feat}/{len(df_out)} rows')
    print(f'Feature columns: {ELEM_FEAT_NAMES}')

    df_out.to_csv(args.output, index=False)
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
