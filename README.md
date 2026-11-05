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
