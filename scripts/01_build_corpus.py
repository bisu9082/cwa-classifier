"""
Script 01: Corpus construction with CWC-grounded labelling
==========================================================
Builds the similarity-expanded training corpus, the external holdout set and
the out-of-scope probe set from the structurally verified compound
registries produced by `00_resolve_verify_cids.py`.

Steps
-----
 1. PubChem 2D fingerprint similarity search (Tanimoto >= threshold) around
    every verified seed CID.
 2. Retrieval of eight physicochemical descriptors plus the isomeric SMILES
    and InChIKey for every candidate CID.
 3. Twelve elemental composition features parsed from the molecular formula.
 4. CWC Schedule matching (`cwc_schedules.py`): per-entry indicators, the
    Schedule level, and the scaffold-motif fingerprint block.
 5. Three labelling regimes, written as separate columns so that the effect
    of label calibration can be quantified directly:

      label_L0  as-submitted: seed label inherited, then overridden by the
                element-composition heuristics of the original pipeline
      label_L1  CWC-calibrated: element heuristics removed; a structural
                match to any Schedule 1 general formula or listed chemical
                is the primary mandatory criterion for the CWA class, and
                all other compounds retain their curated or inherited class
      label_L2  CWC-calibrated and scaffold-filtered: L1 plus removal of
                expanded analogues whose core structure is not consistent
                with the parent scaffolds of their assigned class

    Schedule 2 and 3 membership is deliberately *not* mapped onto the CWA
    class: those Schedules contain dual-use precursors and industrial
    chemicals (thiodiglycol, triethanolamine, dimethyl methylphosphonate)
    whose functional class is not CWA.  Schedule 2/3 status is carried as a
    descriptor instead, so the model can use it without it being asserted
    as a label.

 6. Removal of any holdout or probe CID that also occurs in the training
    corpus, so the external evaluation is strictly out-of-corpus.

Outputs (all under --outdir)
----------------------------
  corpus.csv          expanded training corpus, all feature blocks, L0/L1/L2
  holdout.csv         external holdout, same feature columns
  probe.csv           out-of-scope probe set (organophosphorus flame retardants)
  provenance.json     PubChem query timestamp, endpoints and parameters
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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cwc_schedules as cwc  # noqa: E402

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PROP_LIST = ("MolecularFormula,MolecularWeight,XLogP,TPSA,HBondDonorCount,"
             "HBondAcceptorCount,RotatableBondCount,HeavyAtomCount,"
             "Complexity,SMILES,InChIKey,Title")
SLEEP = 0.25

PHYS = ["mw", "xlogp", "tpsa", "hbd", "hba", "rbc", "hac", "complexity"]
ELEM = ["has_P", "has_F", "has_S", "has_N", "has_Cl", "has_As",
        "has_PF", "has_PS", "has_SN", "P_ratio", "halogen_cnt", "S_cnt"]
CWC_FEATS = cwc.CWC_FEATURE_NAMES

CLASS_NAMES = {0: "CWA", 1: "Simulant", 2: "Pesticide"}


# ── PubChem access ─────────────────────────────────────────────────────────
def _get(url: str, timeout: int = 30, tries: int = 3):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as fh:
                return json.load(fh)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(1.0 * (k + 1))
        except Exception:
            time.sleep(1.0 * (k + 1))
    return None


def similar_cids(seed_cid: int, threshold: int, max_records: int) -> list[int]:
    url = (f"{PUG}/compound/fastsimilarity_2d/cid/{seed_cid}/cids/JSON"
           f"?Threshold={threshold}&MaxRecords={max_records}")
    d = _get(url)
    time.sleep(SLEEP)
    if not d:
        return []
    return [c for c in d.get("IdentifierList", {}).get("CID", []) if c != seed_cid]


def fetch_props(cids: list[int]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for i in range(0, len(cids), 100):
        chunk = ",".join(str(c) for c in cids[i : i + 100])
        d = _get(f"{PUG}/compound/cid/{chunk}/property/{PROP_LIST}/JSON")
        time.sleep(SLEEP)
        if d:
            for p in d.get("PropertyTable", {}).get("Properties", []):
                out[int(p["CID"])] = p
        print(f"    properties {min(i + 100, len(cids))}/{len(cids)}", flush=True)
    return out


# ── feature construction ───────────────────────────────────────────────────
_TOK = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(formula: str) -> dict[str, int]:
    if not formula or not isinstance(formula, str):
        return {}
    counts: dict[str, int] = {}
    core = re.sub(r"[+-]\d*$", "", formula.split(".")[0]).strip()
    for elem, cnt in _TOK.findall(core):
        counts[elem] = counts.get(elem, 0) + (int(cnt) if cnt else 1)
    return counts


def elemental_features(formula: str) -> dict[str, float]:
    e = parse_formula(formula)
    heavy = sum(v for k, v in e.items() if k != "H") or 1
    has_P = int(e.get("P", 0) > 0)
    has_F = int(e.get("F", 0) > 0)
    has_S = int(e.get("S", 0) > 0)
    has_N = int(e.get("N", 0) > 0)
    has_Cl = int(e.get("Cl", 0) > 0)
    has_As = int(e.get("As", 0) > 0)
    return {
        "has_P": has_P, "has_F": has_F, "has_S": has_S, "has_N": has_N,
        "has_Cl": has_Cl, "has_As": has_As,
        "has_PF": int(has_P and has_F),
        "has_PS": int(has_P and has_S),
        "has_SN": int(has_S and has_N),
        "P_ratio": e.get("P", 0) / heavy,
        "halogen_cnt": (e.get("F", 0) + e.get("Cl", 0)
                        + e.get("Br", 0) + e.get("I", 0)),
        "S_cnt": e.get("S", 0),
    }


def phys_features(p: dict) -> dict[str, float] | None:
    def num(key, cast=float):
        v = p.get(key)
        if v in (None, ""):
            return None
        try:
            return cast(v)
        except (TypeError, ValueError):
            return None

    vals = {
        "mw": num("MolecularWeight"),
        "xlogp": num("XLogP"),
        "tpsa": num("TPSA"),
        "hbd": num("HBondDonorCount", int),
        "hba": num("HBondAcceptorCount", int),
        "rbc": num("RotatableBondCount", int),
        "hac": num("HeavyAtomCount", int),
        "complexity": num("Complexity"),
    }
    # molecular weight and heavy-atom count are indispensable; XLogP and TPSA
    # are absent for some inorganic records and are imputed as 0 as in the
    # original pipeline, which is recorded in the provenance file.
    if vals["mw"] is None or vals["hac"] is None:
        return None
    for k in ("xlogp", "tpsa", "hbd", "hba", "rbc", "complexity"):
        if vals[k] is None:
            vals[k] = 0.0
    return vals


# ── labelling regimes ──────────────────────────────────────────────────────
def label_L0(elem: dict, mw: float, inherited: int) -> int:
    """As-submitted element-composition override rules (Table S4 of the
    original submission), reproduced verbatim for the comparison."""
    if elem["has_P"] and elem["has_F"] and mw < 350:
        return 0
    if elem["has_As"]:
        return 0
    if elem["has_S"] and elem["has_Cl"] and not elem["has_P"] and mw < 300:
        return 0
    return inherited


def label_L1(cwc_res: dict, inherited: int) -> int:
    """CWC-calibrated labels.

    A structural match to any Schedule 1 entry is the primary mandatory
    criterion for the CWA class.  Schedule 2 and 3 matches do not force a
    label, because those Schedules are populated by dual-use precursors and
    industrial chemicals whose functional class is not CWA.
    """
    if cwc_res["schedule"] == 1:
        return 0
    return inherited


ELEMENT_SIGNATURE = ["has_P", "has_S", "has_N", "has_Cl", "has_F", "has_As"]


def scaffold_consistent(row: dict, seed_row: dict) -> bool:
    """CWC scaffold-consistency filter for similarity-expanded analogues.

    The Annex defines each class of regulated chemical by a parent scaffold.
    An analogue is therefore retained only if it preserves at least one of
    the parent scaffolds carried by the seed compound it was retrieved for;
    an analogue that has lost every parent scaffold of its seed is a
    redundant neighbour with a non-compliant core structure and is removed.

    Any compound matching a Schedule 1 entry is retained unconditionally,
    since Schedule 1 membership is the primary mandatory criterion.

    Seeds whose parent scaffold is not expressible as a substructure motif
    (the small reactive Schedule 3A chemicals: hydrogen cyanide, phosgene,
    diphosgene) fall back to requiring an identical heavy-element signature.
    """
    if row["cwc_schedule"] == 1:
        return True
    seed_motifs = {m for m in cwc.MOTIF_NAMES if seed_row.get(m, 0)}
    if seed_motifs:
        return any(row.get(m, 0) for m in seed_motifs)
    return all(row.get(e, 0) == seed_row.get(e, 0) for e in ELEMENT_SIGNATURE)


# ── main ───────────────────────────────────────────────────────────────────
def featurise(cid: int, p: dict, inherited: int, is_seed: bool,
              seed_cid: int, meta: dict) -> dict | None:
    phys = phys_features(p)
    if phys is None:
        return None
    formula = p.get("MolecularFormula", "")
    smiles = p.get("SMILES", "")
    elem = elemental_features(formula)
    res = cwc.classify(smiles)
    if not res["parsed"]:
        return None
    cfeat = cwc.schedule_features(smiles)

    row: dict = {
        "cid": cid,
        "seed_cid": seed_cid,
        "is_seed": int(is_seed),
        "name": meta.get("name", ""),
        "subclass": meta.get("subclass", ""),
        "pubchem_title": p.get("Title", ""),
        "formula": formula,
        "smiles": smiles,
        "inchikey": p.get("InChIKey", ""),
        "inherited_label": inherited,
        "cwc_schedule": res["schedule"],
        "cwc_entries": ";".join(res["entries"]),
        "cwc_exempt": res["exempt"],
    }
    row.update(phys)
    row.update(elem)
    row.update(cfeat)
    row["label_L0"] = label_L0(elem, phys["mw"], inherited)
    row["label_L1"] = label_L1(res, inherited)
    row["label_L2"] = row["label_L1"]
    row["scaffold_ok"] = 1          # assigned in a second pass over the corpus
    return row


def apply_scaffold_filter(rows: list[dict]) -> int:
    """Second pass: mark analogues whose core structure left its seed family."""
    seed_rows = {r["seed_cid"]: r for r in rows if r["is_seed"]}
    dropped = 0
    for r in rows:
        if r["is_seed"]:
            r["scaffold_ok"] = 1
            continue
        sr = seed_rows.get(r["seed_cid"])
        ok = 1 if sr is None else int(scaffold_consistent(r, sr))
        r["scaffold_ok"] = ok
        dropped += 1 - ok
    return dropped


FIELDS = (["cid", "seed_cid", "is_seed", "name", "subclass", "pubchem_title",
           "formula", "smiles", "inchikey", "inherited_label",
           "cwc_schedule", "cwc_entries", "cwc_exempt"]
          + PHYS + ELEM + CWC_FEATS
          + ["label_L0", "label_L1", "label_L2", "scaffold_ok"])


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def build_reference_set(registry: Path, provenance: dict,
                        tag: str) -> list[dict]:
    """Featurise a curated set (holdout or probe) without expansion."""
    entries = list(csv.DictReader(open(registry, encoding="utf-8")))
    cids = [int(r["cid"]) for r in entries]
    props = fetch_props(cids)
    provenance[f"{tag}_n_requested"] = len(cids)
    rows = []
    for r in entries:
        cid = int(r["cid"])
        p = props.get(cid)
        if not p:
            print(f"    !! no properties for {r['name']} (CID {cid})")
            continue
        lab = int(r["class_id"])
        row = featurise(cid, p, lab, True, cid, r)
        if row is None:
            print(f"    !! featurisation failed for {r['name']} (CID {cid})")
            continue
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="data/build/seed_registry.csv")
    ap.add_argument("--holdout", default="data/build/holdout_registry.csv")
    ap.add_argument("--probe", default="data/build/probe_registry.csv")
    ap.add_argument("--outdir", default="data/build")
    ap.add_argument("--threshold", type=int, default=85)
    ap.add_argument("--max-records", type=int, default=50)
    a = ap.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    provenance = {
        "query_timestamp_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "pubchem_note": ("PubChem is continuously updated and does not expose "
                         "a discrete release identifier through PUG REST; the "
                         "retrieval timestamp above is the version anchor."),
        "similarity_endpoint": (f"{PUG}/compound/fastsimilarity_2d/cid/"
                                "{CID}/cids/JSON"),
        "similarity_parameters": {"Threshold": a.threshold,
                                  "MaxRecords": a.max_records,
                                  "fingerprint": "PubChem 2D subgraph "
                                                 "fingerprint (881 bit)"},
        "property_endpoint": f"{PUG}/compound/cid/{{CIDs}}/property/{PROP_LIST}/JSON",
        "property_list": PROP_LIST.split(","),
        "property_batch_size": 100,
        "request_interval_s": SLEEP,
        "missing_value_policy": ("MolecularWeight and HeavyAtomCount are "
                                 "required; XLogP, TPSA, HBD, HBA, "
                                 "RotatableBondCount and Complexity are "
                                 "imputed as 0 when absent"),
        "cwc_unencoded_entries": cwc.UNENCODED_ENTRIES,
        "rdkit_version": __import__("rdkit").__version__,
    }

    seeds = list(csv.DictReader(open(a.seeds, encoding="utf-8")))
    print(f"  seeds: {len(seeds)}")

    # ── similarity expansion ──────────────────────────────────────────────
    print("  similarity expansion ...")
    seed_of: dict[int, int] = {}
    inherit: dict[int, int] = {}
    per_seed: dict[int, int] = {}
    for i, s in enumerate(seeds, 1):
        scid = int(s["cid"])
        lab = int(s["class_id"])
        seed_of.setdefault(scid, scid)
        inherit[scid] = lab
        sims = similar_cids(scid, a.threshold, a.max_records)
        per_seed[scid] = len(sims)
        for c in sims:
            if c not in seed_of:
                seed_of[c] = scid
                inherit[c] = lab
        if i % 10 == 0 or i == len(seeds):
            print(f"    {i}/{len(seeds)} seeds, {len(seed_of)} unique CIDs",
                  flush=True)

    provenance["n_seeds"] = len(seeds)
    provenance["n_candidate_cids"] = len(seed_of)
    provenance["analogues_per_seed_mean"] = (
        sum(per_seed.values()) / max(len(per_seed), 1))

    print(f"  retrieving properties for {len(seed_of)} CIDs ...")
    props = fetch_props(sorted(seed_of))

    seed_meta = {int(s["cid"]): s for s in seeds}
    rows = []
    for cid, scid in sorted(seed_of.items()):
        p = props.get(cid)
        if not p:
            continue
        is_seed = cid in seed_meta
        meta = seed_meta.get(cid, {"name": "", "subclass": ""})
        row = featurise(cid, p, inherit[cid], is_seed, scid, meta)
        if row is not None:
            rows.append(row)

    n_dropped = apply_scaffold_filter(rows)
    print(f"  featurised: {len(rows)}  (scaffold filter removes {n_dropped})")

    # ── external sets, with strict CID disjointness ───────────────────────
    print("  external holdout ...")
    hold = build_reference_set(Path(a.holdout), provenance, "holdout")
    print("  out-of-scope probe ...")
    probe = build_reference_set(Path(a.probe), provenance, "probe")

    corpus_cids = {r["cid"] for r in rows}
    hold_drop = [r for r in hold if r["cid"] in corpus_cids]
    probe_drop = [r for r in probe if r["cid"] in corpus_cids]
    hold = [r for r in hold if r["cid"] not in corpus_cids]
    probe = [r for r in probe if r["cid"] not in corpus_cids]
    provenance["holdout_removed_for_corpus_overlap"] = [
        {"cid": r["cid"], "name": r["name"]} for r in hold_drop]
    provenance["probe_removed_for_corpus_overlap"] = [
        {"cid": r["cid"], "name": r["name"]} for r in probe_drop]
    if hold_drop:
        print(f"    removed {len(hold_drop)} holdout compounds present in corpus:")
        for r in hold_drop:
            print(f"      {r['name']} (CID {r['cid']})")
    if probe_drop:
        print(f"    removed {len(probe_drop)} probe compounds present in corpus")

    write_csv(out / "corpus.csv", rows)
    write_csv(out / "holdout.csv", hold)
    write_csv(out / "probe.csv", probe)

    def dist(rs, key):
        d = defaultdict(int)
        for r in rs:
            d[r[key]] += 1
        return {CLASS_NAMES.get(k, k): v for k, v in sorted(d.items())}

    provenance["corpus_n"] = len(rows)
    provenance["corpus_label_L0"] = dist(rows, "label_L0")
    provenance["corpus_label_L1"] = dist(rows, "label_L1")
    provenance["corpus_label_L2_retained"] = dist(
        [r for r in rows if r["scaffold_ok"]], "label_L2")
    provenance["holdout_n"] = len(hold)
    provenance["holdout_dist"] = dist(hold, "label_L1")
    provenance["probe_n"] = len(probe)

    json.dump(provenance, open(out / "provenance.json", "w"), indent=2)

    print(f"\n  corpus            : {len(rows)}")
    print(f"    L0 labels       : {provenance['corpus_label_L0']}")
    print(f"    L1 labels       : {provenance['corpus_label_L1']}")
    print(f"    L2 retained     : {provenance['corpus_label_L2_retained']}"
          f"  (dropped {len(rows) - sum(r['scaffold_ok'] for r in rows)})")
    print(f"  holdout           : {len(hold)}  {provenance['holdout_dist']}")
    print(f"  probe             : {len(probe)}")
    n_flip = sum(1 for r in rows if r["label_L0"] != r["label_L1"])
    print(f"  L0 vs L1 label disagreements: {n_flip} "
          f"({100 * n_flip / max(len(rows), 1):.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
