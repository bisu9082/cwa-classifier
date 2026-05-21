# cwa-classifier

**Quantifying Cross-Family Leakage in Similarity-Expanded Chemometric Classification: A Three-Regime Validation Benchmark for Chemical Warfare Agents, Simulants, and Organophosphate Pesticides**

*Chemometrics and Intelligent Laboratory Systems* (2026, under review)

---

## Overview

This repository provides the full reproducible pipeline for the three-regime chemometric benchmark described in the paper. The core contribution is a **seed-grouped cross-validation** scheme that eliminates within-cluster data leakage in similarity-expanded corpora, and a **pilot external holdout** on 20 OPCW/EPA-registered compounds held entirely outside the training corpus.

### Key Results

| Validation Regime | Best Classifier | Accuracy | Notes |
|---|---|---|---|
| Compound-level CV | RF | 91.5% (95% CI: 91.1–91.9%) | Standard stratified; leakage-susceptible |
| Seed-grouped CV | XGB | 48.9% (95% CI: 45.4–52.2%) | Leakage-free; 44.5 pp below compound-level RF |
| External holdout | SVM | 45.0% | Below 55.0% majority-class baseline |

**Cross-family leakage gap: 44.5 pp (RF) / 42.1 pp (XGB)**

---

## Repository Structure

```
cwa-classifier/
├── data/
│   └── seed_compounds.csv      # 59 curated seed compounds (Table S1)
├── scripts/
│   ├── 01_pubchem_collect.py   # PubChem REST API property retrieval
│   ├── 02_elemental_features.py# 12 elemental composition features
│   ├── 03_similarity_expansion.py # Tanimoto similarity expansion
│   └── 04_classifier_cv.py     # 5-classifier benchmark + CV pipeline
├── requirements.txt
└── README.md
```

---

## Dataset

- **59 unique seed compounds**: 18 CWA, 17 Simulant, 24 Pesticide (see `data/seed_compounds.csv`)
- **Expanded dataset**: 2,502 compounds via PubChem 2D similarity search (Tanimoto ≥ 0.85, max 50/seed)
  - CWA: 778 | Simulant: 734 | Pesticide: 990
- **External holdout**: 20 OPCW-grounded / EPA-registered compounds (not included in repo per arms-control considerations; available upon reasonable request)

---

## Feature Set (20-Dimensional)

**8 Physicochemical** (PubChem REST API):
`MW, XLogP, TPSA, HBD, HBA, RBC, HAC, Complexity`

**12 Elemental Composition** (parsed from molecular formula):
`has_P, has_F, has_S, has_N, has_Cl, has_As, has_PF, has_PS, has_SN, P_ratio, halogen_cnt, S_cnt`

---

## Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Collect PubChem properties for seed compounds
```bash
python scripts/01_pubchem_collect.py \
    --input  data/seed_compounds.csv \
    --output data/seed_properties.csv
```

### 3. Add elemental composition features
```bash
python scripts/02_elemental_features.py \
    --input  data/seed_properties.csv \
    --output data/seed_features.csv
```

### 4. Expand dataset via similarity search
```bash
python scripts/03_similarity_expansion.py \
    --input      data/seed_features.csv \
    --output     data/expanded_dataset.csv \
    --threshold  85 \
    --max-records 50
```
> Note: Exact compound counts depend on PubChem database state at query time. The paper reports 2,502 compounds (queried 2025-11).

### 5. Run classifier benchmark
```bash
python scripts/04_classifier_cv.py \
    --input   data/expanded_dataset.csv \
    --holdout data/external_holdout.csv \
    --outdir  results
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

---

## Validation Design

**Compound-level CV**: Standard stratified 5-fold × 5 random seeds (25 evaluations).
Within-seed-family analogs can appear in both training and test folds → inflated accuracy.

**Seed-grouped CV**: All compounds derived from the same seed CID are co-assigned to the same fold. Eliminates within-cluster structural leakage. Each of 59 seed CIDs is cyclically assigned to one of 5 folds, shuffled per random seed.

**External holdout**: 20 structurally novel compounds from OPCW schedules and EPA pesticide registries, held completely outside the training corpus.

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

This code is released under the MIT License.
Data derived from PubChem is subject to [PubChem's data policy](https://pubchem.ncbi.nlm.nih.gov/docs/data-sources).
