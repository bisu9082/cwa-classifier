"""
Expand the out-of-scope probe (Minor-comment 1) with additional out-of-domain
industrial chemicals, reusing the exact feature pipeline of 01_build_corpus.py
and the CWC matcher, so the new rows are column-identical to data/build/probe.csv.

Categories (all non-CWA, non-pesticide, non-simulant industrial chemicals):
  * additional organophosphorus flame retardants / plasticisers
  * phthalate and non-phosphorus plasticisers
  * unrelated common industrial organics, solvents and pharmaceuticals
"""
from __future__ import annotations
import sys, time, json, urllib.request, urllib.parse
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import importlib.util as ilu
spec = ilu.spec_from_file_location("bc", HERE / "01_build_corpus.py")
bc = ilu.module_from_spec(spec); spec.loader.exec_module(bc)
import cwc_schedules as cwc

ROOT = HERE.parent
BUILD = ROOT / "data" / "build"
PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PROP = ("MolecularFormula,MolecularWeight,XLogP,TPSA,HBondDonorCount,"
        "HBondAcceptorCount,RotatableBondCount,HeavyAtomCount,"
        "Complexity,SMILES,InChIKey,Title")

# curated out-of-domain industrial chemicals (name, subclass)
WANT = [
    ("Tri-n-butyl phosphate", "Organophosphorus flame retardant"),
    ("Tricresyl phosphate", "Organophosphorus flame retardant"),
    ("Tris(2-butoxyethyl) phosphate", "Organophosphorus flame retardant"),
    ("Tris(2-ethylhexyl) phosphate", "Organophosphorus flame retardant"),
    ("2-Ethylhexyl diphenyl phosphate", "Organophosphorus flame retardant"),
    ("Isodecyl diphenyl phosphate", "Organophosphorus flame retardant"),
    ("Resorcinol bis(diphenyl phosphate)", "Organophosphorus flame retardant"),
    ("Trixylyl phosphate", "Organophosphorus flame retardant"),
    ("Cresyl diphenyl phosphate", "Organophosphorus flame retardant"),
    ("Bis(2-ethylhexyl) phthalate", "Phthalate plasticiser"),
    ("Dibutyl phthalate", "Phthalate plasticiser"),
    ("Diethyl phthalate", "Phthalate plasticiser"),
    ("Dimethyl phthalate", "Phthalate plasticiser"),
    ("Benzyl butyl phthalate", "Phthalate plasticiser"),
    ("Diisononyl phthalate", "Phthalate plasticiser"),
    ("Bisphenol A", "Industrial organic"),
    ("Toluene", "Industrial solvent"),
    ("Phenol", "Industrial organic"),
    ("Aniline", "Industrial organic"),
    ("Naphthalene", "Industrial organic"),
    ("Styrene", "Industrial organic"),
    ("Benzophenone", "Industrial organic"),
    ("Triclosan", "Industrial organic"),
    ("Melamine", "Industrial organic"),
    ("Vanillin", "Industrial organic"),
    ("Caffeine", "Pharmaceutical / consumer"),
    ("Acetaminophen", "Pharmaceutical / consumer"),
    ("Ibuprofen", "Pharmaceutical / consumer"),
    ("Aspirin", "Pharmaceutical / consumer"),
    ("Nicotine", "Pharmaceutical / consumer"),
    ("Citric acid", "Industrial organic"),
    ("Urea", "Industrial organic"),
    ("Glycerol", "Industrial organic"),
    ("Caprolactam", "Industrial organic"),
]


def _get(url, tries=3):
    for _ in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=30) as fh:
                return json.loads(fh.read())
        except Exception:
            time.sleep(0.6)
    return None


def fetch_by_name(name):
    u = f"{PUG}/compound/name/{urllib.parse.quote(name)}/property/{PROP}/JSON"
    d = _get(u)
    time.sleep(0.2)
    if not d:
        return None
    props = d.get("PropertyTable", {}).get("Properties", [])
    return props[0] if props else None


def main():
    tmpl = pd.read_csv(BUILD / "probe.csv")
    cols = list(tmpl.columns)
    existing_ik = set(tmpl["inchikey"].astype(str))
    rows = []
    for name, sub in WANT:
        p = fetch_by_name(name)
        if not p:
            print(f"  MISS {name}"); continue
        formula = p.get("MolecularFormula", "")
        smiles = p.get("SMILES", "") or p.get("ConnectivitySMILES", "")
        ik = p.get("InChIKey", "")
        if ik in existing_ik:
            print(f"  dup  {name} ({ik})"); continue
        phys = bc.phys_features(p)
        if phys is None:
            print(f"  nophys {name}"); continue
        elem = bc.elemental_features(formula)
        cres = cwc.classify(smiles)
        feats = cwc.schedule_features(smiles)
        row = {c: 0 for c in cols}
        row.update({
            "cid": int(p["CID"]), "seed_cid": int(p["CID"]), "is_seed": 0,
            "name": name, "subclass": sub, "pubchem_title": p.get("Title", ""),
            "formula": formula, "smiles": smiles, "inchikey": ik,
            "inherited_label": 3, "cwc_schedule": cres["schedule"],
            "cwc_entries": ";".join(cres["entries"]), "cwc_exempt": cres["exempt"],
            "label_L0": 3, "label_L1": 3, "label_L2": 3, "scaffold_ok": 1,
        })
        row.update(phys); row.update(elem); row.update(feats)
        rows.append(row); existing_ik.add(ik)
        print(f"  OK   {name:38s} {formula:14s} sch={cres['schedule']}")

    new = pd.DataFrame(rows)[cols]
    out = pd.concat([tmpl, new], ignore_index=True)
    out.to_csv(BUILD / "probe.csv", index=False)
    # keep a copy of the pre-expansion probe for provenance
    tmpl.to_csv(BUILD / "probe_v1_original10.csv", index=False)
    print(f"\nprobe: {len(tmpl)} -> {len(out)} compounds "
          f"({(out.cwc_schedule>0).sum()} CWC-scheduled, "
          f"{(out.cwc_schedule==0).sum()} unscheduled)")


if __name__ == "__main__":
    main()
