# Spatiotemporal Deep Learning for EEG-Based Emotion Recognition

Case Study 2. M.Sc. Applied Data Science and Analytics, SRH Hochschule Heidelberg.
Supervisor: Prof. Dr. Binh Vu.

Binary valence and arousal classification on **DEAP**, evaluated under strict
32-fold leave-one-subject-out (LOSO) cross-validation, with **SEED** used as an
external cross-dataset test set. The study progresses from classical feature-based
baselines through deep spatiotemporal architectures to unsupervised domain
adaptation.

---

## Key findings

Two evaluation settings are reported throughout. **Within-DEAP** trains and tests
on DEAP only, holding out one subject at a time. **Cross-dataset** trains on DEAP
and tests on SEED, a different lab with different subjects and different recording
hardware, which is the harder and more realistic setting.

Rows marked **This work** are results produced here; the others are published
numbers on the identical task, included for comparison.

### Cross-dataset: DEAP to SEED, binary, balanced accuracy

| Method | Source | Result |
|---|---|---|
| GPTDS + PC-TTA | Imtiaz & Khan, 2024 | 67.44% |
| **Best technique combination** | **This work** | **66.09%** |
| **Ensembling + test-time augmentation, no target data** | **This work** | **66.02%** |
| **Zero-shot, no adaptation** | **This work** | **64.55%** |
| Baseline without adaptation | Imtiaz & Khan, 2024 | 60.35% |
| Pre-trained, no adaptation | Imtiaz & Khan, 2024 | 54.24% |

**1.35 points from the highest published figure**, which is within the split-to-split
variation observed here, across 448 cross-dataset experiments.

### Within-DEAP: leave-one-subject-out, balanced accuracy

| Method | Source | Valence | Arousal |
|---|---|---|---|
| HEDN | Published, 2025 | 73.07% | 73.20% |
| DBPM | Published, 2025 | 70.65% | 71.20% |
| DANN | Published, 2018 | 69.20% | 68.68% |
| **CNN-LSTM + attention, subject-level MMD alignment** | **This work** | **65.31%** | **64.42%** |
| **Transformer / graph neural network variants** | **This work** | **63.15%** | **64.34%** |
| ATDD-LSTM | Published, 2020 | 59.06% | 72.97% |
| **Classical baseline, linear SVM on spectral features (per-window)** | **This work** | **53.61%** | **52.30%** |

Protocols differ between rows: the classical baseline is per-window, the deep results
are per-trial. Roughly 8 points of the difference between them is attributable to the
evaluation protocol rather than the model; see the report for the full comparison.

Within DEAP the difference is larger: 7.76 points below the HEDN figure on valence,
though ahead of the 2020 ATDD-LSTM baseline. The two are not measured under an
identical protocol.

### What the experiments showed

1. **Single-split results were not stable.** One configuration reached 68.59% on
   a single data split, but fell to 63.30% ± 1.56 when repeated across multiple
   splits. All numbers reported above are therefore multi-split.
2. **The widely used Kaggle mirror of DEAP has corrupted valence and arousal
   columns.** All labels here are rebuilt from the official
   `participant_ratings.xls`.

Full result tables and the complete analysis are in the case study report, submitted separately.

---

## Repository layout

```
notebooks/
  01_preprocessing/       Label rebuild from official ratings (00_label_rebuild.ipynb),
                          DEAP loading, filtering, epoching
  02_classical_baselines/ DE + PSD features, SVM / Random Forest, optimisation
  03_deep_baselines/      CNN / LSTM / BiLSTM baselines and diagnostics
  04_multi_source_da/     Multi-source domain adaptation, per-trial protocol
  05_reliability_dualnet/ Source reliability weighting and confidence-routed dual head
  06_transformer_gnn/     Transformer + graph neural network, per-trial/per-window
  07_cross_dataset/       SEED cross-dataset transfer, CORAL UDA, ablations
```

---

## Evaluation protocol

- **Cross-validation:** 32-fold leave-one-subject-out on DEAP, one held-out
  subject per fold, no subject appearing in both train and test. The within-DEAP
  deep experiments use a two-way split rather than a nested train/validation/test
  split, so the stopping epoch is the best-scoring epoch within each fold. The
  classical baselines and the cross-dataset results are not affected.
- **Labels:** binary, thresholded at 5.0 on the 1–9 self-assessment scale.
- **Metric:** balanced accuracy, because DEAP's valence and arousal classes are
  imbalanced and plain accuracy flatters a majority-class predictor.
- **Significance:** reported p-values test each model against chance.
- **External validation:** SEED is never used for training. The strict zero-shot results
  (64.55% and 66.02%) use no SEED data at any point and are evaluated on all 15 subjects.
  The multi-split runs additionally use three held-out SEED subjects to select the stopping
  epoch, and are reported separately for that reason.
- **Reproducibility:** random seeds are fixed inside each notebook. Because
  evaluation is 32-fold LOSO rather than a single split, run-to-run variation is
  small but not zero.
