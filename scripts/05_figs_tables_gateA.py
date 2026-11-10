"""Gate-A (Schedule-1-mandatory) honest figures and LaTeX tables."""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import importlib.util as ilu
spec = ilu.spec_from_file_location("g", HERE / "03_gate.py")
G = ilu.module_from_spec(spec); spec.loader.exec_module(G)

ROOT = HERE.parent; OUT = ROOT / "results"; FIG = ROOT / "figures"; BUILD = ROOT / "data" / "build"
P = {"cream": "#F2DFC0", "red": "#C94F4A", "amber": "#E8943A", "gold": "#F0C040",
     "teal": "#4AACB0", "dblue": "#5B8DB8", "brown": "#8B6347", "dteal": "#2E7D7A",
     "coral": "#E8836A", "sage": "#7FAF6A", "tan": "#C9A882", "rose": "#B84040"}
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans",
                     "axes.edgecolor": "#444", "axes.linewidth": 0.8, "figure.dpi": 150})

hold = pd.read_csv(BUILD / "holdout.csv"); probe = pd.read_csv(BUILD / "probe.csv")

def provenance(df, dist, thr):
    """Classify each compound by which gate branch decides it."""
    s = df["cwc_schedule"].values
    ex = (df["cwc_exempt"].astype(str).str.len() > 0).values if "cwc_exempt" in df else np.zeros(len(df), bool)
    cnt = {"Schedule 1 → CWA (mandatory)": 0, "Annex exemption → CWA forbidden": 0,
           "Out-of-domain → unknown": 0, "Classifier decides (Sch. 2/3, unlisted)": 0}
    for i in range(len(df)):
        if s[i] == 1: cnt["Schedule 1 → CWA (mandatory)"] += 1
        elif ex[i]: cnt["Annex exemption → CWA forbidden"] += 1
        elif s[i] == 0 and dist[i] > thr: cnt["Out-of-domain → unknown"] += 1
        else: cnt["Classifier decides (Sch. 2/3, unlisted)"] += 1
    return cnt

corp = pd.read_csv(BUILD / "corpus.csv")
trf = [G.morgan(x) for x in corp["smiles"]]; trf = [f for f in trf if f]
thr = G.ad_threshold(trf, 99)
hod = [G.nn_distance(G.morgan(x), trf) for x in hold["smiles"]]
prd = [G.nn_distance(G.morgan(x), trf) for x in probe["smiles"]]
prov_h = provenance(hold, hod, thr); prov_p = provenance(probe, prd, thr)

# ── Figure 1: decision provenance (where statutory authority binds) ─────────
labels = list(prov_h.keys())
cols = [P["rose"], P["amber"], P["brown"], P["dblue"]]
fig, ax = plt.subplots(figsize=(10.5, 3.2))
for yi, (name, prov, tot) in enumerate([("External holdout\n(n=72)", prov_h, 72),
                                        ("Out-of-scope probe\n(n=40)", prov_p, 40)]):
    left = 0
    for lab, c in zip(labels, cols):
        v = prov[lab]
        if v:
            ax.barh(yi, v, left=left, color=c, edgecolor="w")
            ax.text(left + v/2, yi, str(v), ha="center", va="center",
                    color="white", fontsize=9.5, weight="bold")
        left += v
ax.set_yticks([0, 1]); ax.set_yticklabels(["External holdout\n(n=72)", "Out-of-scope probe\n(n=40)"])
ax.set_xlabel("number of compounds"); ax.set_xlim(0, 74)
ax.spines[["top", "right"]].set_visible(False)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cols]
ax.legend(handles, labels, frameon=False, fontsize=8.3, ncol=2,
          loc="upper center", bbox_to_anchor=(0.5, -0.28))
ax.set_title("Where the statutory gate binds: hard rules act only where they are sound; "
             "Schedule 2/3 is left to the classifier", fontsize=10.8, weight="bold", pad=8)
fig.tight_layout(); fig.savefig(FIG / "Fig_gate_provenance.png", bbox_inches="tight"); plt.close(fig)

# ── Figure 2: workflow (Gate-A) ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 3.2)); ax.axis("off")
ax.set_xlim(0, 11); ax.set_ylim(0, 3)
def box(x, y, w, h, txt, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                 fc=c, ec="#444", lw=1.1))
    ax.text(x+w/2, y+h/2, txt, ha="center", va="center", fontsize=9, weight="bold")
def arr(x1, x2, y):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>",
                 mutation_scale=13, color="#555", lw=1.3))
box(0.05, 1.1, 1.35, 0.9, "candidate\nstructure", P["cream"])
box(1.75, 1.1, 1.5, 0.9, "A3 out-of-scope\nguard", P["tan"])
box(3.55, 1.1, 1.6, 0.9, "CWC rule engine\nSchedule 1?", P["gold"])
box(5.45, 1.1, 1.6, 0.9, "Annex\nexemption?", P["coral"])
box(7.35, 1.1, 1.5, 0.9, "AD reject\n(unlisted)", P["sage"])
box(9.15, 1.1, 1.7, 0.9, "classifier\nSch.2/3, unlisted", P["dblue"])
for x1, x2 in [(1.4, 1.75), (3.25, 3.55), (5.15, 5.45), (7.05, 7.35), (8.85, 9.15)]:
    arr(x1, x2, 1.55)
ax.text(2.5, 0.62, "match\n→ unknown", ha="center", fontsize=7.6, color=P["rose"])
ax.text(4.35, 0.62, "yes → CWA\n(mandatory)", ha="center", fontsize=7.6, color=P["rose"])
ax.text(6.25, 0.62, "yes → not CWA", ha="center", fontsize=7.6, color=P["rose"])
ax.text(8.1, 0.62, "far → unknown", ha="center", fontsize=7.6, color=P["rose"])
ax.text(10.0, 0.62, "CWA/Sim/Pest", ha="center", fontsize=7.6, color=P["dteal"])
ax.set_title("Inspection-oriented screening workflow (Gate-A): statutory rules bind on "
             "Schedule 1 and exemptions; the classifier resolves the rest",
             fontsize=10.5, weight="bold", pad=6)
fig.savefig(FIG / "Fig_gate_workflow.png", bbox_inches="tight"); plt.close(fig)

# ── Table 1: honest gate benchmark (F3, all classifiers) ────────────────────
det = json.load(open(OUT / "gate_details.json"))
rows = [r for r in det["rows"] if r["scheme"] == "F3"]
lines = [r"\begin{tabular}{lccccc}", r"\toprule",
         r"Classifier & CWA recall & CWA recall & Acc.\ on & Coverage & False-CWA on probe \\",
         r" & (ML) & (gated) & predicted & (gated) & ML$\to$gate \\", r"\midrule"]
for r in rows:
    lines.append(f"{r['clf']} & {r['ho_CWArecall_ML']:.0f}\\% & {r['ho_CWArecall_GATE']:.0f}\\% & "
                 f"{r['ho_acc_on_predicted_GATE']:.1f}\\% & {r['ho_coverage_GATE']:.0f}\\% & "
                 f"{r['probe_falseCWA_ML_%']:.0f}\\%$\\to${r['probe_falseCWA_GATE_%']:.0f}\\% \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "table_gate_benchmark.tex").write_text("\n".join(lines))

# ── Table 2: where the gate binds (counts) ──────────────────────────────────
tl = [r"\begin{tabular}{lcc}", r"\toprule",
      r"Gate branch & Holdout (n=72) & Probe (n=40) \\", r"\midrule"]
for lab in labels:
    tl.append(f"{lab.replace(chr(0x2192),'$\\to$')} & {prov_h[lab]} & {prov_p[lab]} \\\\")
tl += [r"\bottomrule", r"\end{tabular}"]
(OUT / "table_gate_bindings.tex").write_text("\n".join(tl))

print("prov holdout:", prov_h)
print("prov probe  :", prov_p)
print("figs:", [p.name for p in FIG.glob("Fig_gate_*.png")])
