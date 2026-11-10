"""
Post-hoc statutory hard-rule gate (E1-strong) over the ML multiclass head.
=========================================================================
Implements the revision's Major-1 response: the CWC rule engine is given
decision authority *after* the ML classifier, instead of being one more
soft input feature.

Gate pipeline, applied per test compound:
  1. A3 out-of-scope guard   saxitoxin / ricin / quaternary-carbamate
                             skeletons  ->  UNKNOWN (routed out, never a class)
  2. B3 applicability-domain Tanimoto NN distance to the training corpus
                             above the 95th-percentile threshold -> UNKNOWN
  3. statutory hard gate     CWC schedule >= 1  ->  CWA   (ML overruled)
                             CWC schedule == 0  ->  not CWA; the ML head only
                             decides Simulant vs Pesticide among unscheduled
                             structures
Outputs before/after metrics for the external holdout and the OOD probe.

Run from repo root:  python -I scripts/03_gate.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cwc_schedules as cwc                               # noqa: E402
import importlib.util as _ilu                             # noqa: E402
_spec = _ilu.spec_from_file_location("bench02", HERE / "02_benchmark.py")
_bench = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_bench)
SCHEMES, make_clf, CLASS_NAMES = _bench.SCHEMES, _bench.make_clf, _bench.CLASS_NAMES

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             matthews_corrcoef, f1_score, confusion_matrix,
                             precision_recall_fscore_support)

ROOT = HERE.parent
BUILD = ROOT / "data" / "build"
OUT = ROOT / "results"
UNKNOWN = -1                      # reject / out-of-scope label
CWA, SIM, PEST = 0, 1, 2

# ── A3 out-of-scope recognisers (the UNENCODED_ENTRIES, as guards) ──────────
# Saxitoxin connectivity skeleton (InChIKey first block) and a SMARTS for the
# Schedule-1A.16 quaternary / bis-quaternary carbamate family.  Ricin is a
# protein and never arrives as a small-molecule SMILES.
_SAXITOXIN_SKEL = "RPQXVSUAYFXFJA"          # saxitoxin InChIKey block-1
_CARBAMATE_Q = Chem.MolFromSmarts("[NX4+][#6][OX2]C(=O)N")   # quaternary carbamate


def a3_out_of_scope(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return False
    try:
        block1 = Chem.InchiToInchiKey(Chem.MolToInchi(mol)).split("-")[0]
        if block1 == _SAXITOXIN_SKEL:
            return True
    except Exception:
        pass
    if _CARBAMATE_Q is not None and mol.HasSubstructMatch(_CARBAMATE_Q):
        return True
    return False


# ── B3 applicability domain: Tanimoto NN distance ──────────────────────────
def morgan(smiles: str):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def nn_distance(fp, train_fps):
    if fp is None:
        return 1.0
    sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
    return 1.0 - max(sims) if sims else 1.0


def ad_threshold(train_fps, pct=95):
    """95th-percentile of leave-one-out NN distance within the training set."""
    d = []
    for i, fp in enumerate(train_fps):
        others = train_fps[:i] + train_fps[i + 1:]
        sims = DataStructs.BulkTanimotoSimilarity(fp, others)
        d.append(1.0 - max(sims) if sims else 1.0)
    return float(np.percentile(d, pct))


# ── gate ───────────────────────────────────────────────────────────────────
def apply_gate(ml_pred, scores, schedules, smiles, nn_dists, thr, exempt=None):
    """Gate-A: statutory authority bounded to where it is sound.

    Only a Schedule-1 match is a *mandatory* CWA criterion; an explicit Annex
    exemption *forbids* the regulated class; Schedule 2/3 and unlisted
    structures are left to the classifier, consistent with the Schedule-vs-class
    orthogonality analysis. This matches the reviewer's own mixture-of-experts
    description (Schedule 1 -> CWA; ambiguous Schedule 2/3 -> ML; no match ->
    unknown via the applicability domain)."""
    if exempt is None:
        exempt = [False] * len(ml_pred)
    out = []
    for p, s, sch, smi, dist, ex in zip(ml_pred, scores, schedules, smiles,
                                        nn_dists, exempt):
        # 1. out-of-scope structural guard
        if a3_out_of_scope(smi):
            out.append(UNKNOWN); continue
        # 2. Schedule 1 is the only mandatory CWA criterion (sound: no
        #    simulant or pesticide matches Schedule 1).
        if sch == 1:
            out.append(CWA); continue
        # 3. an explicit Annex exemption forbids the regulated class.
        if ex and p == CWA:
            out.append(SIM if s[SIM] >= s[PEST] else PEST); continue
        # 4. unscheduled & out-of-domain -> reject to unknown.
        if sch == 0 and dist > thr:
            out.append(UNKNOWN); continue
        # 5. otherwise the classifier decides (Schedule 2/3 dual-use
        #    precursors and unlisted structures, incl. unlisted agents).
        out.append(int(p))
    return np.array(out)


def metrics(y_true, y_pred, tag):
    mask_known = y_pred != UNKNOWN
    n = len(y_true)
    n_unknown = int((~mask_known).sum())
    yt, yp = y_true[mask_known], y_pred[mask_known]
    acc_all = float((y_pred == y_true).mean()) * 100          # unknown counts wrong
    if len(yt):
        acc_known = accuracy_score(yt, yp) * 100
        bacc = balanced_accuracy_score(yt, yp)
        mcc = matthews_corrcoef(yt, yp) if len(set(yt)) > 1 else float("nan")
        mf1 = f1_score(yt, yp, average="macro", zero_division=0)
        cm = confusion_matrix(y_true, y_pred,
                              labels=[CWA, SIM, PEST]).tolist()
    else:
        acc_known = bacc = mcc = mf1 = float("nan"); cm = []
    return {"tag": tag, "n": n, "n_unknown": n_unknown,
            "acc_incl_unknown": round(acc_all, 2),
            "acc_on_predicted": round(acc_known, 2),
            "balanced_acc": round(bacc, 4) if bacc == bacc else None,
            "mcc": round(mcc, 4) if mcc == mcc else None,
            "macro_f1": round(mf1, 4) if mf1 == mf1 else None,
            "confusion": cm}


def main():
    corp = pd.read_csv(BUILD / "corpus.csv")
    hold = pd.read_csv(BUILD / "holdout.csv")
    probe = pd.read_csv(BUILD / "probe.csv")

    ytr = corp["label_L2"].astype(int).values
    yho = hold["label_L2"].astype(int).values
    # probe is out-of-scope: it has no true in-domain class (label 3). Correct
    # behaviour is UNKNOWN; we score it as a false-alarm rate instead.

    print("Building Morgan fingerprints + AD threshold ...", flush=True)
    tr_fps = [morgan(s) for s in corp["smiles"]]
    tr_fps = [f for f in tr_fps if f is not None]
    thr = ad_threshold(tr_fps, 99)
    print(f"  AD threshold (99th pct Tanimoto distance) = {thr:.3f}", flush=True)

    ho_fps = [morgan(s) for s in hold["smiles"]]
    pr_fps = [morgan(s) for s in probe["smiles"]]
    ho_dist = [nn_distance(f, tr_fps) for f in ho_fps]
    pr_dist = [nn_distance(f, tr_fps) for f in pr_fps]
    print(f"  holdout NN dist  median={np.median(ho_dist):.3f}  "
          f"frac>thr={np.mean(np.array(ho_dist) > thr):.2f}", flush=True)
    print(f"  probe   NN dist  median={np.median(pr_dist):.3f}  "
          f"frac>thr={np.mean(np.array(pr_dist) > thr):.2f}", flush=True)

    # A3 coverage audit
    ho_a3 = sum(a3_out_of_scope(s) for s in hold["smiles"])
    pr_a3 = sum(a3_out_of_scope(s) for s in probe["smiles"])
    cp_a3 = sum(a3_out_of_scope(s) for s in corp["smiles"])
    print(f"  A3 out-of-scope matches  corpus={cp_a3} holdout={ho_a3} probe={pr_a3}",
          flush=True)

    rows = []
    for scheme in ["F2", "F3"]:
        cols = SCHEMES[scheme]
        Xtr = corp[cols].astype(float).values
        Xho = hold[cols].astype(float).values
        Xpr = probe[cols].astype(float).values
        for clf_name in ["DT", "SVM", "LR", "RF", "XGB"]:
            clf = make_clf(clf_name, proba=True)
            clf.fit(Xtr, ytr)
            # ML-only
            yp_ho = clf.predict(Xho)
            S_ho = clf.predict_proba(Xho)
            yp_pr = clf.predict(Xpr)
            S_pr = clf.predict_proba(Xpr)
            # gated
            ho_ex = (hold["cwc_exempt"].astype(str).str.len() > 0).values \
                if "cwc_exempt" in hold else None
            pr_ex = (probe["cwc_exempt"].astype(str).str.len() > 0).values \
                if "cwc_exempt" in probe else None
            g_ho = apply_gate(yp_ho, S_ho, hold["cwc_schedule"].values,
                              hold["smiles"].values, ho_dist, thr, ho_ex)
            g_pr = apply_gate(yp_pr, S_pr, probe["cwc_schedule"].values,
                              probe["smiles"].values, pr_dist, thr, pr_ex)

            m_before = metrics(yho, yp_ho, f"{scheme}/{clf_name} holdout ML-only")
            m_after = metrics(yho, g_ho, f"{scheme}/{clf_name} holdout GATED")
            # safety-critical: CWA recall (did we catch every real agent?)
            cwa_mask = yho == CWA
            rec_before = float((yp_ho[cwa_mask] == CWA).mean()) * 100
            rec_after = float((g_ho[cwa_mask] == CWA).mean()) * 100
            # coverage = fraction of holdout the gate still predicts
            cov = float((g_ho != UNKNOWN).mean()) * 100
            # accuracy on the compounds actually predicted (coverage-adjusted)
            pm = g_ho != UNKNOWN
            acc_cov = float((g_ho[pm] == yho[pm]).mean()) * 100 if pm.any() else 0
            # probe: the consequential error is a FALSE CWA alarm on OOD chemicals
            falsecwa_before = float((yp_pr == CWA).mean()) * 100
            falsecwa_after = float((g_pr == CWA).mean()) * 100
            probe_abstain = float((g_pr == UNKNOWN).mean()) * 100
            rows.append({
                "scheme": scheme, "clf": clf_name,
                "ho_CWArecall_ML": round(rec_before, 1),
                "ho_CWArecall_GATE": round(rec_after, 1),
                "ho_acc_on_predicted_GATE": round(acc_cov, 1),
                "ho_coverage_GATE": round(cov, 1),
                "ho_mcc_ML": m_before["mcc"], "ho_mcc_GATE": m_after["mcc"],
                "probe_falseCWA_ML_%": round(falsecwa_before, 1),
                "probe_falseCWA_GATE_%": round(falsecwa_after, 1),
                "probe_abstain_GATE_%": round(probe_abstain, 1),
                "cm_ML": m_before["confusion"], "cm_GATE": m_after["confusion"],
            })
            print(f"{scheme}/{clf_name:3s}  CWA-recall {rec_before:5.1f}->{rec_after:5.1f}"
                  f"  acc@cov {acc_cov:5.1f} (cov {cov:4.0f}%)"
                  f"  probe falseCWA {falsecwa_before:4.0f}->{falsecwa_after:4.0f}%",
                  flush=True)

    df = pd.DataFrame(rows)
    df.drop(columns=["cm_ML", "cm_GATE"]).to_csv(OUT / "gate_benchmark.csv",
                                                 index=False)
    json.dump({"ad_threshold": thr, "rows": rows},
              open(OUT / "gate_details.json", "w"), indent=2)
    print("\nwrote results/gate_benchmark.csv + gate_details.json")


if __name__ == "__main__":
    main()
