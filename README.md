# ECGLab - Classical ML Module

Beat-level arrhythmia classification on MIT-BIH (AAMI classes N, S, V, F, Q) with
classical ML, using the inter-patient split of de Chazal et al. (DS1 train / DS2 test).

## Structure
- `src/preprocessing.py`: band-pass 0.5-40 Hz + z-score, beat segmentation, AAMI mapping, DS1/DS2 split, RR features
- `src/features.py`: 12 features (4 RR + QRS width, R amplitude, ST deviation, statistics)
- `src/evaluate.py`: unified metrics (accuracy, precision, recall, F1, AUC, confusion matrix, ROC)
- `src/rpeaks.py`: simplified Pan-Tompkins R-peak detector (median-based threshold)
- `reports/`: figures (SHAP, filters, evaluation) and result tables

## Setup
Install dependencies with `pip install -r requirements.txt`.
Data: MIT-BIH via `wfdb.dl_database('mitdb', DIR)` (not included in the repo).

## Results (DS2 test set, inter-patient)
| Model | Macro F1 (N,S,V,F) | Macro AUC (N,S,V,F) | S recall | F recall |
|---|---|---|---|---|
| SVM | 0.549 | 0.917 | 0.385 | 0.206 |
| RF + SMOTE | 0.642 | 0.903 | 0.300 | 0.528 |
| XGBoost | 0.610 | 0.913 | 0.292 | 0.428 |

R-peak detection (tolerance 150 ms, all 44 records):
| Method | Se | PPV | F1 |
|---|---|---|---|
| neurokit2 | 0.994 | 0.995 | 0.994 |
| Ours v2 (median threshold) | 0.985 | 0.990 | 0.987 |
| Ours v1 (max threshold) | 0.934 | 0.996 | 0.964 |

## Notes and limitations
- Class Q has 8 train / 7 test beats: its metrics are not meaningful and are excluded from macro averages.
- Class F in DS1 is concentrated in one patient (record 208: 372 of 414 beats), so F metrics are indicative only.
- Wavelet coefficients of the raw beat were tested and hurt generalization across patients (patient-specific morphology).
- Stacking with patient-grouped CV performed worse than RF alone (too few patients, F concentrated in one record).
- R-peak thresholds were tuned on the same records used for evaluation (detector has 2 parameters and no training).
- Balancing (undersampling N, SMOTE for S and F) is applied on the training set only.

## Additional experiments
- **15-feature variant:** the 12 features plus 3 RR features normalized by the median of the previous 20 intervals of the same record (causal, usable in real time). Chosen with patient-grouped CV on DS1 before looking at DS2. On DS2 it improves V (RF F1 0.918 to 0.940) and leaves S unchanged excluding record 232 (`reports/results_15feat_vs_v1.csv`).
- **Dominant records:** on DS2, record 232 holds 75% of S beats and record 213 holds 93% of F beats, so S and F metrics mostly reflect these patients. Results excluding them are in `reports/results_excluding_dominant_records.csv` and `reports/per_record_S_F.csv`.
- **Negative results:** wavelet features, segment features, a second lead, stacking, a 1D-CNN (macro F1 0.49 on DS2) and the same CNN with augmentation (0.48) all failed to beat RF (`reports/cnn1d_summary_3seeds.csv`). With 22 training records, generalization across patients is the limit.

## External validation (INCART, never used for any decision)
Models trained on MIT-BIH DS1 only; INCART resampled 257 to 360 Hz, lead II, same pipeline. See `reports/external_incart.csv`.

| Model | Macro F1 (N,S,V,F) | F1 V | V recall | F1 S (all / without I33,I34) |
|---|---|---|---|---|
| RF v1 | 0.553 | 0.838 | 0.924 | 0.394 / 0.184 |
| RF 15 feat | 0.558 | 0.873 | 0.921 | 0.366 / 0.168 |
| XGBoost v1 | 0.575 | 0.857 | 0.903 | 0.458 / 0.228 |
| XGBoost 15 feat | 0.577 | 0.875 | 0.902 | 0.447 / 0.240 |

S recall is high (0.78-0.85) but precision is low, and S is concentrated in records I33 and I34. F did not generalize (F1 about 0.03, 219 beats).

## Honest notes
- The 15-feature set was selected by grouped CV on DS1. The choice of RF as the reference model was made after seeing DS2 results, so INCART is the first evaluation on which no decision was taken.
- Code for the CNN and INCART experiments is not yet in this repository.
