# cwa-classifier

**Quantifying Cross-Family Leakage in Similarity-Expanded Chemometric Classification: A Three-Regime Validation Benchmark for Chemical Warfare Agents, Simulants, and Organophosphate Pesticides**

*Chemometrics and Intelligent Laboratory Systems* (2026, under review)

---

## Important note on this repository (2026-09 update)

The version of this pipeline originally released alongside the submitted
manuscript contained a data-curation error: 37 of the 59 PubChem compound
identifiers in the submitted seed list resolved to chemically unrelated
compounds, and the error propagated into the similarity-expanded corpus and
the external holdout. This repository now reflects the **corrected**
pipeline used in the revised manuscript: every compound identifier is
resolved from its CAS registry number and verified against an independently
curated reference molecular formula before being admitted to the dataset
(`scripts/00_resolve_verify_cids.py`). Entries that fail verification are
excluded rather than used. The full resolution/verification audit trail is
included under `data/build/`.

On the corrected data, the paper's central conclusion changes: models
trained on similarity-expanded corpora **do** generalise above chance to
structurally distant compounds. See Key Results below.

---

## Overview

This repository provides the reproducible pipeline for the three-regime
chemometric benchmark described in the paper: a **seed-grouped
cross-validation** scheme that eliminates within-cluster data leakage in
similarity-expanded corpora, CWC Annex Schedule structural matching encoded
as executable substructure queries, and an **external holdout** of 72
OPCW/EPA-registered compounds held entirely outside the training corpus.

### Key Results (verified data)

| Validation Regime | Best Classifier | Accuracy | Notes |
|---|---|---|---|
| Compound-level CV | XGB | 95.5% | Standard stratified; leakage-susceptible |
| Seed-grouped CV | XGB | 82.2% $\pm$ 7.9% | Leakage-free |
| External holdout | DT / XGB, RF | 88.9% / 81.9% | Above the majority-class baseline (37.5%) at $p<0.001$ |

**Leakage gap (compound-level vs. seed-grouped): 11.0–15.5 pp**

---

## Repository Structure

```
cwa-classifier/
├── data/
│   ├── seed_compounds.csv          # 105 curated seed compounds (name, CAS, class, formula)
│   ├── external_holdout.csv        # 72 curated external holdout compounds
│   ├── probe_flame_retardants.csv  # 10 out-of-scope probe compounds
│   └── build/                      # outputs of the resolve/verify and corpus-build steps
│       ├── seed_registry.csv           # resolved + verified CID, formula, SMILES for every seed
│       ├── holdout_registry.csv        # same, for the external holdout
│       ├── probe_registry.csv          # same, for the probe set
│       ├── provenance.json             # PubChem query parameters, timestamps, corpus counts
│       ├── constant_features.json      # CWC feature columns constant in the corpus
│       ├── corpus.csv                  # full similarity-expanded corpus (3,307 compounds, all descriptors)
│       ├── holdout.csv                 # resolved external holdout with all descriptors (72)
│       └── probe.csv                   # resolved probe set with all descriptors (10)
├── scripts/
│   ├── 00_resolve_verify_cids.py   # CAS/name -> CID resolution + molecular-formula verification
│   ├── 01_build_corpus.py          # similarity expansion, descriptors, CWC labelling regimes
│   ├── 01b_refresh_cwc_features.py # recompute CWC Schedule/scaffold features in place
│   ├── 02_benchmark.py             # three-regime, three-label, three-feature-scheme benchmark
│   ├── cwc_schedules.py            # CWC Annex Schedule 1/2/3 structural matcher (SMARTS-based)
│   └── test_cwc_schedules.py       # 84-case validation suite for the Schedule matcher
├── results/                        # benchmark.csv and diagnostic outputs of 02_benchmark.py
├── requirements.txt
└── README.md
```

---

## Dataset

- **105 verified seed compounds**: 42 CWA, 24 Simulant, 39 Pesticide
  (`data/seed_compounds.csv`; resolved and verified in `data/build/seed_registry.csv`)
- **Similarity-expanded corpus**: 3,307 compounds via PubChem 2D fingerprint
  similarity search (Tanimoto $\geq$ 0.85, max 50/seed), reduced to 2,567 by
  a parent-scaffold consistency filter (`data/build/corpus.csv`)
- **External holdout**: 72 OPCW-grounded / EPA-registered compounds, held
  entirely outside the training corpus (25 CWA, 20 Simulant, 27 Pesticide)
- **Out-of-scope probe**: 10 organophosphorus flame retardants, reported as
  an applicability-domain check rather than a fourth class

## CWC Annex Schedule matching

`scripts/cwc_schedules.py` encodes fourteen Annex general-formula entries and
42 listed chemicals across Schedules 1–3 as executable structural queries
(RDKit SMARTS for the parent scaffold plus programmatic substituent-count
checks), rather than the element-presence heuristics (P/F/As/S) used in the
originally submitted version. The matcher is validated against 84 documented
cases in `scripts/test_cwc_schedules.py`.

## Feature Set

**8 Physicochemical** (PubChem REST API): `MW, XLogP, TPSA, HBD, HBA, RBC, HAC, Complexity`

**12 Elemental composition** (parsed from molecular formula): `has_P, has_F, has_S, has_N, has_Cl, has_As, has_PF, has_PS, has_SN, P_ratio, halogen_cnt, S_cnt`

**44 CWC-derived** (from `cwc_schedules.py`): 18 Schedule-entry/level indicators plus 26 scaffold-motif fingerprints

---

## Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Resolve and verify compound identifiers
```bash
python scripts/00_resolve_verify_cids.py \
    --input  data/seed_compounds.csv \
    --outdir data/build \
    --registry seed_registry.csv \
    --report   seed_resolution_report.csv
# repeat for data/external_holdout.csv and data/probe_flame_retardants.csv
```

### 3. Build the similarity-expanded corpus
```bash
python scripts/01_build_corpus.py \
    --seeds   data/build/seed_registry.csv \
    --holdout data/build/holdout_registry.csv \
    --probe   data/build/probe_registry.csv \
    --outdir  data/build \
    --threshold 85 --max-records 50
```
> Note: exact compound counts depend on PubChem database state at query
> time. The paper reports 3,307 compounds (queried 2026-09).

### 4. Run the benchmark
```bash
python scripts/02_benchmark.py \
    --corpus  data/build/corpus.csv \
    --holdout data/build/holdout.csv \
    --probe   data/build/probe.csv \
    --outdir  results
```

### 5. Run the CWC matcher test suite
```bash
python scripts/test_cwc_schedules.py
```

---

## Classifiers

| Branch | Algorithm | Key Hyperparameters |
|---|---|---|
| DT | Decision Tree | max_depth=5, min_samples_leaf=4 |
| SVM | SVM (linear) | C=1.0, kernel=linear |
| LR | Logistic Regression | C=1.0, max_iter=1000 |
| RF | Random Forest | n_estimators=200 |
| XGB | XGBoost | n_estimators=200, max_depth=5, lr=0.1 |

## Validation Design

**Compound-level CV**: Standard stratified 5-fold x 5 random seeds.
Within-seed-family analogues can appear in both training and test folds,
inflating accuracy.

**Seed-grouped CV**: All compounds derived from the same seed CID are
co-assigned to the same fold, eliminating within-cluster structural leakage.

**External holdout**: 72 structurally novel compounds from OPCW schedules
and EPA pesticide registries, held completely outside the training corpus.

**Out-of-scope probe**: 10 organophosphorus flame retardants outside all
three classes, used as an applicability-domain check.

---

## Citation

```bibtex
@article{Lee2026CILSLeakage,
  author  = {Lee, Doo-hee and Nam, HanYeol and Kim, Jeongyun and
             Yoo, Jin and Jang, Yoon Jeong and Kim, Min-Kun and
             Kang, Ku and Kim, Sungbong},
  title   = {Quantifying Cross-Family Leakage in Similarity-Expanded
             Chemometric Classification: A Three-Regime Validation
             Benchmark for Chemical Warfare Agents, Simulants, and
             Organophosphate Pesticides},
  journal = {Chemometrics and Intelligent Laboratory Systems},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

This code is released under the MIT License (see `LICENSE`).
Data derived from PubChem is subject to [PubChem's data policy](https://pubchem.ncbi.nlm.nih.gov/docs/data-sources).
