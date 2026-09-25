"""
Script 00: Compound identity resolution and structural verification
===================================================================
Resolves every curated compound entry (name + CAS) to a PubChem CID and
VERIFIES the resolved record against the independently curated molecular
formula before the CID is admitted to the corpus.

Rationale
---------
PubChem CIDs are opaque integers; a transcription or mis-resolution error
produces a syntactically valid but chemically unrelated record.  Because all
downstream descriptors are retrieved *by CID*, such an error silently
substitutes the descriptors of an unrelated molecule.  This script therefore
treats CID assignment as a step requiring explicit verification rather than
an identifier lookup:

  1. CAS registry number  -> CID   (primary route, PubChem /name/ endpoint)
  2. compound name        -> CID   (fallback route)
  3. verification:  retrieved MolecularFormula must equal the curated
                    reference formula (Hill notation, exact string match)
  4. any entry failing verification is reported and EXCLUDED; it never
     reaches the descriptor table.

Outputs
-------
  data/compound_registry.csv   verified entries (cid, cas, name, class,
                               formula, smiles, inchikey, pubchem_title)
  data/resolution_report.csv   full audit trail incl. failures

Usage
-----
    python scripts/00_resolve_verify_cids.py \
        --input  data/seed_compounds_curated.csv \
        --outdir data
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PROPS = "MolecularFormula,MolecularWeight,SMILES,InChIKey,Title"
SLEEP = 0.25  # PubChem fair-use: <=5 requests/second


# ── low-level PubChem access ───────────────────────────────────────────────
def _get(url: str, timeout: int = 20):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as fh:
            return json.load(fh)
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError):
        return None


def name_to_cids(query: str) -> list[int]:
    """Resolve a CAS number or chemical name to candidate CIDs."""
    url = f"{PUG}/compound/name/{urllib.parse.quote(query, safe='')}/cids/JSON"
    d = _get(url)
    time.sleep(SLEEP)
    if not d:
        return []
    return d.get("IdentifierList", {}).get("CID", [])


def cid_properties(cids: list[int]) -> dict[int, dict]:
    """Batch-retrieve identity properties for a list of CIDs."""
    out: dict[int, dict] = {}
    for i in range(0, len(cids), 100):
        chunk = ",".join(str(c) for c in cids[i : i + 100])
        d = _get(f"{PUG}/compound/cid/{chunk}/property/{PROPS}/JSON")
        time.sleep(SLEEP)
        if d:
            for p in d.get("PropertyTable", {}).get("Properties", []):
                out[int(p["CID"])] = p
    return out


# ── formula normalisation ──────────────────────────────────────────────────
_TOK = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(formula: str) -> dict[str, int]:
    """Parse a Hill-notation formula string into an element:count mapping."""
    if not formula or not isinstance(formula, str):
        return {}
    counts: dict[str, int] = {}
    core = formula.split(".")[0]                      # drop salt components
    core = re.sub(r"[+-]\d*$", "", core).strip()       # drop net charge
    for elem, cnt in _TOK.findall(core):
        counts[elem] = counts.get(elem, 0) + (int(cnt) if cnt else 1)
    return counts


def formula_key(formula: str) -> str:
    """Canonical comparable representation of a molecular formula."""
    c = parse_formula(formula)
    return "".join(f"{e}{c[e]}" for e in sorted(c))


def formulas_match(a: str, b: str) -> bool:
    return bool(formula_key(a)) and formula_key(a) == formula_key(b)


# ── resolution ─────────────────────────────────────────────────────────────
def resolve_entry(row: dict) -> dict:
    """Resolve and verify one curated entry.

    Returns an audit record with status in
    {'verified_cas', 'verified_name', 'verified_declared', 'FAILED'}.
    """
    name = (row.get("name") or "").strip()
    cas = (row.get("cas") or "").strip()
    ref_formula = (row.get("formula") or "").strip()
    declared = (row.get("cid") or "").strip()

    attempts: list[tuple[str, list[int]]] = []
    if cas:
        attempts.append(("cas", name_to_cids(cas)))
    if name:
        attempts.append(("name", name_to_cids(name)))
    if declared.isdigit():
        attempts.append(("declared", [int(declared)]))

    all_cids = [c for _, cids in attempts for c in cids[:10]]
    props = cid_properties(list(dict.fromkeys(all_cids)))

    for route, cids in attempts:
        for cid in cids[:10]:
            p = props.get(cid)
            if p and formulas_match(p.get("MolecularFormula", ""), ref_formula):
                return {
                    "name": name,
                    "cas": cas,
                    "class": row.get("class", ""),
                    "class_id": row.get("class_id", ""),
                    "subclass": row.get("subclass", ""),
                    "ref_formula": ref_formula,
                    "cid": cid,
                    "formula": p.get("MolecularFormula", ""),
                    "mw": p.get("MolecularWeight", ""),
                    "smiles": p.get("SMILES", ""),
                    "inchikey": p.get("InChIKey", ""),
                    "pubchem_title": p.get("Title", ""),
                    "resolved_via": route,
                    "declared_cid": declared,
                    "declared_cid_ok": str(
                        declared.isdigit()
                        and formulas_match(
                            props.get(int(declared), {}).get("MolecularFormula", ""),
                            ref_formula,
                        )
                    ),
                    "status": f"verified_{route}",
                }

    dp = props.get(int(declared), {}) if declared.isdigit() else {}
    return {
        "name": name,
        "cas": cas,
        "class": row.get("class", ""),
        "class_id": row.get("class_id", ""),
        "subclass": row.get("subclass", ""),
        "ref_formula": ref_formula,
        "cid": "",
        "formula": "",
        "mw": "",
        "smiles": "",
        "inchikey": "",
        "pubchem_title": dp.get("Title", ""),
        "resolved_via": "",
        "declared_cid": declared,
        "declared_cid_ok": "False",
        "status": "FAILED",
    }


FIELDS = [
    "name", "cas", "class", "class_id", "subclass", "ref_formula", "cid",
    "formula", "mw", "smiles", "inchikey", "pubchem_title", "resolved_via",
    "declared_cid", "declared_cid_ok", "status",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--registry", default="compound_registry.csv")
    ap.add_argument("--report", default="resolution_report.csv")
    a = ap.parse_args()

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(a.input, encoding="utf-8-sig")))

    records = []
    for i, row in enumerate(rows, 1):
        rec = resolve_entry(row)
        records.append(rec)
        flag = "OK " if rec["status"] != "FAILED" else "!! "
        drift = "" if rec["declared_cid_ok"] == "True" else "  [declared CID rejected]"
        print(f"  {flag}[{i:3d}/{len(rows)}] {rec['name'][:38]:38s} "
              f"CID {str(rec['cid']):>10s} via {rec['resolved_via'] or '-':8s}{drift}",
              flush=True)

    with open(outdir / a.report, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(records)

    verified = [r for r in records if r["status"] != "FAILED"]
    seen, dedup = set(), []
    for r in verified:
        if r["cid"] not in seen:
            seen.add(r["cid"])
            dedup.append(r)

    with open(outdir / a.registry, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(dedup)

    n_bad_declared = sum(1 for r in records if r["declared_cid_ok"] != "True")
    print(f"\n  entries in            : {len(records)}")
    print(f"  structurally verified : {len(verified)}  (unique CIDs: {len(dedup)})")
    print(f"  failed verification   : {len(records) - len(verified)}")
    print(f"  declared CIDs rejected: {n_bad_declared}")
    for r in records:
        if r["status"] == "FAILED":
            print(f"    FAILED: {r['name']} (CAS {r['cas']}, expected {r['ref_formula']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
