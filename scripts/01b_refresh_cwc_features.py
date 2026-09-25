"""
Script 01b: Recompute the CWC descriptor block in place
=======================================================
Recomputes the CWC Schedule assignment and the scaffold-motif fingerprint for
an existing corpus, holdout or probe file from the SMILES already stored in
it, and rewrites the dependent label and filter columns.

This exists so that a correction to the matcher does not require re-querying
PubChem: the compound set and its physicochemical descriptors are unchanged,
only the structural features derived from them are recomputed.  Keeping the
retrieval fixed also keeps the provenance timestamp meaningful.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cwc_schedules as cwc  # noqa: E402

import importlib
b = importlib.import_module("01_build_corpus")


def refresh(path: Path, is_corpus: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    feats = [cwc.schedule_features(s if isinstance(s, str) else "")
             for s in df["smiles"]]
    res = [cwc.classify(s if isinstance(s, str) else "") for s in df["smiles"]]
    for col in cwc.CWC_FEATURE_NAMES:
        df[col] = [f[col] for f in feats]
    df["cwc_schedule"] = [r["schedule"] for r in res]
    df["cwc_entries"] = [";".join(r["entries"]) for r in res]
    df["cwc_exempt"] = [r["exempt"] for r in res]
    df["label_L1"] = [b.label_L1(r, int(inh)) for r, inh
                      in zip(res, df["inherited_label"])]
    df["label_L2"] = df["label_L1"]
    if is_corpus:
        rows = df.to_dict("records")
        dropped = b.apply_scaffold_filter(rows)
        df = pd.DataFrame(rows)[df.columns]
        print(f"    scaffold filter removes {dropped}")
    else:
        df["scaffold_ok"] = 1
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="data/build")
    a = ap.parse_args()
    bld = Path(a.build)
    for name, is_corpus in [("corpus.csv", True), ("holdout.csv", False),
                            ("probe.csv", False)]:
        print(f"  {name}")
        df = refresh(bld / name, is_corpus)
        df.to_csv(bld / name, index=False)
        if is_corpus:
            prov = json.load(open(bld / "provenance.json"))
            names = {0: "CWA", 1: "Simulant", 2: "Pesticide"}
            prov["corpus_label_L1"] = {
                names[k]: int(v) for k, v in
                sorted(df.label_L1.value_counts().to_dict().items())}
            prov["corpus_label_L2_retained"] = {
                names[k]: int(v) for k, v in
                sorted(df[df.scaffold_ok == 1].label_L2
                       .value_counts().to_dict().items())}
            prov["cwc_features_recomputed"] = True
            json.dump(prov, open(bld / "provenance.json", "w"), indent=2)
            print(f"    L1 {prov['corpus_label_L1']}")
            print(f"    L2 {prov['corpus_label_L2_retained']}")
    # report descriptors that carry no variance, so they can be disclosed
    df = pd.read_csv(bld / "corpus.csv")
    hold = pd.read_csv(bld / "holdout.csv")
    const = [c for c in cwc.CWC_FEATURE_NAMES if df[c].nunique() == 1]
    live_in_holdout = [c for c in const if hold[c].nunique() > 1]
    print(f"\n  constant in the training corpus ({len(const)}): "
          f"{', '.join(const) if const else 'none'}")
    print(f"  of those, non-constant in the holdout: "
          f"{', '.join(live_in_holdout) if live_in_holdout else 'none'}")
    json.dump({"constant_in_corpus": const,
               "constant_in_corpus_live_in_holdout": live_in_holdout},
              open(bld / "constant_features.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
