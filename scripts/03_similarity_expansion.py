"""
Script 03: PubChem Structural Similarity Search & Dataset Expansion
===================================================================
Expands the curated seed dataset via PubChem FastSimilarity_2D
(Tanimoto >= 0.85, max 50 similar CIDs per seed).

Auto-labeling rules are applied to resolve label conflicts in expanded
compounds based on elemental composition heuristics (see Table S4 in SI).

Input:  data/seed_features.csv  (59 seed compounds with features)
Output: data/expanded_dataset.csv  (~2,502 compounds)

Usage:
    python scripts/03_similarity_expansion.py \
        --input   data/seed_features.csv \
        --output  data/expanded_dataset.csv \
        --threshold 85 \
        --max-records 50

Note: Exact compound counts depend on PubChem database state at query time.
      The paper reports 2,502 compounds (queried 2025-11).

Reference: Code Listing S3 in Supplementary Information.
"""

import argparse
import time
import re
from collections import defaultdict

import pandas as pd
import requests

from scripts.script_02_elemental_features import (
    formula_to_elem_features, ELEM_FEAT_NAMES, parse_formula
)

PUBCHEM_BASE = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug'
PROPS = (
    'MolecularFormula,MolecularWeight,XLogP,TPSA,'
    'HBondDonorCount,HBondAcceptorCount,'
    'RotatableBondCount,HeavyAtomCount,Complexity'
)


# ── Auto-labeling rules ────────────────────────────────────────────────────

def auto_label(formula: str, mw: float, seed_class_id: int) -> int:
    """Apply structural rule-based class override to expanded compounds.

    Rules (applied in priority order):
      1. has_P AND has_F AND MW < 350  → CWA (G-series)
      2. has_As                         → CWA (arsenical agents)
      3. has_S AND has_Cl AND NOT has_P AND MW < 300 → CWA (mustard vesicants)
      4. C_count <= 3 AND (has_Cl OR has_N) AND NOT has_P AND MW < 120
                                        → CWA (blood/choking agents)
      5. has_P                          → retain seed class
      default                           → retain seed class

    Parameters
    ----------
    formula      : molecular formula string
    mw           : molecular weight (g/mol)
    seed_class_id: class ID inherited from seed (0=CWA, 1=Simulant, 2=Pesticide)

    Returns
    -------
    Resolved class ID (int).
    """
    e = parse_formula(formula)
    has_P  = e.get('P',  0) > 0
    has_F  = e.get('F',  0) > 0
    has_S  = e.get('S',  0) > 0
    has_Cl = e.get('Cl', 0) > 0
    has_N  = e.get('N',  0) > 0
    has_As = e.get('As', 0) > 0
    C_cnt  = e.get('C',  0)

    if has_P and has_F and mw < 350:             return 0   # G-agent
    if has_As:                                    return 0   # arsenical
    if has_S and has_Cl and not has_P and mw < 300: return 0  # mustard
    if C_cnt <= 3 and (has_Cl or has_N) and not has_P and mw < 120:
        return 0   # blood/choking agent
    return seed_class_id   # P-containing or default: keep seed class


def get_similar_cids(seed_cid: int, threshold: int = 85,
                     max_records: int = 50) -> list:
    """Retrieve structurally similar CIDs from PubChem 2D fingerprint search.

    Parameters
    ----------
    seed_cid   : PubChem CID of the query compound
    threshold  : Tanimoto similarity threshold (0–100, default 85)
    max_records: maximum number of similar CIDs to retrieve

    Returns
    -------
    List of similar CIDs (seed itself excluded).
    """
    url = (f'{PUBCHEM_BASE}/compound/fastsimilarity_2d/cid/{seed_cid}'
           f'/cids/JSON?Threshold={threshold}&MaxRecords={max_records}')
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            cids = r.json().get('IdentifierList', {}).get('CID', [])
            return [c for c in cids if c != seed_cid]
    except Exception as exc:
        print(f'  [WARN] CID {seed_cid}: {exc}')
    return []


def fetch_props_single(cid: int) -> dict | None:
    """Retrieve 8 physicochemical properties for one CID."""
    url = f'{PUBCHEM_BASE}/compound/cid/{cid}/property/{PROPS}/JSON'
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            p = r.json()['PropertyTable']['Properties'][0]
            return {
                'cid':        int(cid),
                'formula':    p.get('MolecularFormula', ''),
                'mw':         float(p.get('MolecularWeight', 0) or 0),
                'xlogp':      float(p.get('XLogP', 0) or 0),
                'tpsa':       float(p.get('TPSA', 0) or 0),
                'hbd':        int(p.get('HBondDonorCount', 0) or 0),
                'hba':        int(p.get('HBondAcceptorCount', 0) or 0),
                'rbc':        int(p.get('RotatableBondCount', 0) or 0),
                'hac':        int(p.get('HeavyAtomCount', 0) or 0),
                'complexity': float(p.get('Complexity', 0) or 0),
            }
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Expand seed dataset via PubChem 2D similarity search.')
    parser.add_argument('--input',       default='data/seed_features.csv')
    parser.add_argument('--output',      default='data/expanded_dataset.csv')
    parser.add_argument('--threshold',   type=int, default=85,
                        help='Tanimoto threshold 0-100 (default: 85)')
    parser.add_argument('--max-records', type=int, default=50,
                        help='Max similar CIDs per seed (default: 50)')
    parser.add_argument('--rate-limit',  type=float, default=1.0,
                        help='Seconds to sleep between seed queries (default: 1.0)')
    args = parser.parse_args()

    seeds = pd.read_csv(args.input)
    print(f'Loaded {len(seeds)} seed compounds')

    CLASS_MAP = {0: 'CWA', 1: 'Simulant', 2: 'Pesticide'}

    # ── Step 1: collect similar CIDs for each seed ─────────────────────
    candidate_cids = {}   # cid -> seed_class_id (first-seen seed class)
    for _, row in seeds.iterrows():
        seed_cid = int(row['cid'])
        seed_class_id = int(row['class_id'])
        similar = get_similar_cids(seed_cid, args.threshold, args.max_records)
        for c in similar:
            if c not in candidate_cids:
                candidate_cids[c] = (seed_cid, seed_class_id)
        print(f'  Seed CID {seed_cid:>12} ({row["class"]:>9}) → '
              f'{len(similar):3d} similar | total candidates: {len(candidate_cids)}',
              flush=True)
        time.sleep(args.rate_limit)

    # Remove seed CIDs from candidates
    seed_cid_set = set(seeds['cid'].tolist())
    candidate_cids = {c: v for c, v in candidate_cids.items()
                      if c not in seed_cid_set}
    print(f'\nUnique non-seed candidates: {len(candidate_cids)}')

    # ── Step 2: retrieve properties for each candidate ─────────────────
    rows = []
    for i, (cid, (seed_cid, seed_class_id)) in enumerate(candidate_cids.items(), 1):
        props = fetch_props_single(cid)
        if props is None:
            continue
        formula = props['formula']
        mw      = props['mw']
        class_id = auto_label(formula, mw, seed_class_id)
        feat = formula_to_elem_features(formula)
        row = {
            'cid':      cid,
            'seed_cid': seed_cid,
            'class_id': class_id,
            'class':    CLASS_MAP[class_id],
            'is_seed':  0,
            **{k: v for k, v in props.items() if k != 'cid'},
            **dict(zip(ELEM_FEAT_NAMES, feat)),
        }
        rows.append(row)
        if i % 100 == 0:
            print(f'  [{i}/{len(candidate_cids)}] retrieved so far: {len(rows)}',
                  flush=True)
        time.sleep(0.2)

    # ── Step 3: combine seeds + expanded ───────────────────────────────
    expanded_df = pd.DataFrame(rows)

    # Seed rows: add is_seed=1 and seed_cid=self
    seeds_out = seeds.copy()
    seeds_out['is_seed']  = 1
    seeds_out['seed_cid'] = seeds_out['cid']

    combined = pd.concat([seeds_out, expanded_df], ignore_index=True)
    combined.to_csv(args.output, index=False)

    print(f'\n=== Expansion complete ===')
    print(f'Seeds:         {len(seeds_out)}')
    print(f'Expanded:      {len(expanded_df)}')
    print(f'Total:         {len(combined)}')
    print(combined.groupby('class')['cid'].count().to_string())
    print(f'\nSaved to {args.output}')


if __name__ == '__main__':
    main()
