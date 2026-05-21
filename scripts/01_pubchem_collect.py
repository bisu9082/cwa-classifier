"""
Script 01: PubChem Molecular Property Collection
=================================================
Retrieves 8 physicochemical descriptors for each seed compound
via the PubChem REST API (CID-based lookup).

Output: data/seed_properties.csv

Usage:
    python scripts/01_pubchem_collect.py \
        --input  data/seed_compounds.csv \
        --output data/seed_properties.csv

Reference: Code Listing S1 in Supplementary Information.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

# ── PubChem property string ────────────────────────────────────────────────
PROPS = (
    'MolecularFormula,MolecularWeight,XLogP,TPSA,'
    'HBondDonorCount,HBondAcceptorCount,'
    'RotatableBondCount,HeavyAtomCount,Complexity'
)
PUBCHEM_BASE = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug'


def fetch_props(cid: int) -> tuple:
    """Retrieve 8 physicochemical properties for a single PubChem CID.

    Returns
    -------
    (cid, dict) on success, (cid, None) on failure.
    """
    url = f'{PUBCHEM_BASE}/compound/cid/{cid}/property/{PROPS}/JSON'
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            p = r.json()['PropertyTable']['Properties'][0]
            return int(cid), {
                'cid':        int(cid),
                'formula':    p.get('MolecularFormula'),
                'mw':         float(p.get('MolecularWeight', 0) or 0),
                'xlogp':      float(p.get('XLogP', 0) or 0),
                'tpsa':       float(p.get('TPSA', 0) or 0),
                'hbd':        int(p.get('HBondDonorCount', 0) or 0),
                'hba':        int(p.get('HBondAcceptorCount', 0) or 0),
                'rbc':        int(p.get('RotatableBondCount', 0) or 0),
                'hac':        int(p.get('HeavyAtomCount', 0) or 0),
                'complexity': float(p.get('Complexity', 0) or 0),
            }
    except Exception as exc:
        print(f'  [WARN] CID {cid}: {exc}')
    return int(cid), None


def collect_batch(cid_list: list, batch_size: int = 100,
                  max_workers: int = 8) -> dict:
    """Batch-collect properties with ThreadPoolExecutor.

    Parameters
    ----------
    cid_list   : list of PubChem CIDs
    batch_size : number of CIDs per concurrent batch
    max_workers: number of parallel threads

    Returns
    -------
    dict mapping cid -> property dict
    """
    results = {}
    for i in range(0, len(cid_list), batch_size):
        batch = cid_list[i:i + batch_size]
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(fetch_props, c): c for c in batch}
            for f in as_completed(futures):
                cid, props = f.result()
                if props:
                    results[cid] = props
        n_done = min(i + batch_size, len(cid_list))
        print(f'  [{n_done}/{len(cid_list)}] collected: {len(results)}',
              flush=True)
        time.sleep(0.3)   # ~1 req/sec effective rate limit
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Collect PubChem physicochemical descriptors for seed compounds.')
    parser.add_argument('--input',  default='data/seed_compounds.csv',
                        help='Seed compound CSV (must have cid, class, class_id columns)')
    parser.add_argument('--output', default='data/seed_properties.csv',
                        help='Output CSV with 8 physicochemical descriptors')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel threads (default: 8)')
    args = parser.parse_args()

    # Load seed compounds
    seeds = pd.read_csv(args.input)
    print(f'Loaded {len(seeds)} seed compounds from {args.input}')

    cid_list = seeds['cid'].tolist()
    print(f'Retrieving PubChem properties for {len(cid_list)} CIDs ...')

    results = collect_batch(cid_list, max_workers=args.workers)
    print(f'\nSuccessfully retrieved: {len(results)}/{len(cid_list)} CIDs')

    # Merge with class labels
    prop_df = pd.DataFrame(results.values())
    merged = seeds[['name', 'cid', 'class', 'class_id']].merge(
        prop_df, on='cid', how='left')

    merged.to_csv(args.output, index=False)
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
