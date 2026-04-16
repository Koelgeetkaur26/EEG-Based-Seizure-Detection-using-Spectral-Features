# EEG-Based Seizure Detection

Automated detection of epileptic seizures from scalp EEG recordings using spectral band power analysis, phase-amplitude coupling, and classical machine learning. Built on real clinical data from the CHB-MIT Scalp EEG Database (PhysioNet).

---

## Overview

Epileptic seizures produce characteristic shifts in EEG activity across frequency bands — elevated theta and gamma power, suppressed alpha, increased inter-channel synchrony, and critically, a coupling between the phase of slow theta oscillations and the amplitude of fast gamma bursts. This project extracts these neurophysiological signatures from 4-second EEG epochs and trains classifiers to distinguish ictal (seizure) from interictal (non-seizure) states, then localises seizure onset in time using a sliding-window detection system.

The pipeline runs end-to-end in a single Google Colab notebook, from raw `.edf` download to temporal onset detection with clinically relevant latency.

---

## Pipeline

```
Raw EEG (.edf)  —  CHB-MIT, subject chb01, file chb01_03
        │
        ▼
Preprocessing
  ├── Bandpass filter     (0.5–50 Hz, zero-phase Butterworth)
  ├── Notch filter        (50 Hz power-line removal)
  └── Common average reference
        │
        ▼
Epoch Extraction
  ├── 4-second non-overlapping windows
  ├── Labels: 1 = ictal, 0 = interictal
  └── Peri-ictal buffer excluded (±10s around seizure boundaries)
        │
        ▼
Feature Extraction  (per epoch, 26 features total)
  ├── Spectral band power  — delta, theta, alpha, beta, gamma  (Welch PSD)
  ├── Hjorth Complexity    — frequency variability over time
  ├── Inter-channel correlation — hypersynchrony measure
  ├── Theta-gamma PAC      — Modulation Index (Canolty et al., 2006)
  └── Supporting: spectral entropy, SEF95, variance, kurtosis, skewness, ZCR
        │
        ▼
Class Imbalance Handling
  └── SMOTE oversampling (Chawla et al., 2002)
        │
        ▼
Classification  —  5-fold stratified cross-validation
  ├── Random Forest
  ├── Gradient Boosting
  └── SVM (RBF kernel)
        │
        ▼
Evaluation
  ├── AUC-ROC, F1, Sensitivity, Specificity
  ├── Confusion matrix + ROC curve
  ├── Feature importance (Gini)
  └── Sliding-window temporal onset detection
```

---

## Dataset

**CHB-MIT Scalp EEG Database**
Shoeb, A., & Guttag, J. (2009). *Application of machine learning to epileptic seizure detection.* ICML.
PhysioNet: https://physionet.org/content/chbmit/1.0.0/

- 22 pediatric patients with intractable seizures
- Sampled at 256 Hz, 23-channel standard 10-20 system
- Long-term recordings with clinically annotated seizure onset/offset times
- Freely available (requires free PhysioNet account)

**File used:** `chb01_03.edf` — subject chb01
- Seizure 1: 2996s → 3036s (40 seconds)
- Seizure 2: 3368s → 3426s (58 seconds)

---

## Features

### Why these four feature categories?

**Spectral band power** captures the frequency-domain signature of seizure. During ictal states, theta (4–8 Hz) and gamma (30–50 Hz) power increase while alpha (8–13 Hz) is suppressed — a well-established pattern in the clinical EEG literature.

**Hjorth Complexity** measures how rapidly the dominant frequency changes over time. Normal background EEG has relatively stable frequency content; epileptic discharges are irregular and rapidly shifting, producing high complexity values.

**Inter-channel correlation** captures hypersynchrony. Healthy brain activity is largely desynchronised across regions. Seizures force independent regions into synchronised firing, raising inter-channel correlation sharply — often before changes are visible in any single channel.

**Theta-gamma phase-amplitude coupling (PAC)** directly measures cross-frequency interaction. The Modulation Index (Canolty et al., 2006) quantifies whether gamma amplitude is locked to a preferred phase of the theta cycle. High MI indicates coupling — a hallmark of ictal hypersynchrony. This was the single most discriminative feature in this dataset.

```
MI = |mean( A_gamma(t) × exp(i × φ_theta(t)) )|

where:
  A_gamma  = gamma band amplitude envelope  (30–50 Hz, Hilbert transform)
  φ_theta  = theta band instantaneous phase (4–8 Hz,  Hilbert transform)
```

---

## Results

Evaluated on subject chb01, 5-fold stratified cross-validation on SMOTE-balanced dataset:

| Classifier | AUC-ROC | Sensitivity | Specificity |
|---|---|---|---|
| Random Forest | 0.998 | 0.99 | 0.98 |
| Gradient Boosting | 0.998 | 1.00 | 0.97 |
| SVM (RBF) | 0.997 | 0.99 | 0.97 |

**Sliding-window onset detection:** Detection latency < 5 seconds from true annotated onset (well within the 10-second clinical threshold).

**Top features by importance (Random Forest):**
1. Theta-gamma PAC (0.21)
2. SEF std (0.16)
3. ZCR std (0.11)
4. Gamma power std (0.07)
5. Gamma power mean (0.06)

### Limitation
These results are **patient-specific** (subject chb01 only). Cross-patient generalisation typically yields lower AUC (~0.80–0.90) due to significant variability in seizure morphology between individuals. Patient-specific calibration is standard in clinical EEG systems and would be a natural direction for extension.

---

## Output Figures

| File | Description |
|---|---|
| `raw_eeg_visualization.png` | EEG trace + spectrogram around first seizure |
| `feature_distributions.png` | Ictal vs interictal distributions for 8 key features |
| `model_evaluation.png` | Confusion matrix, ROC curve, classifier comparison |
| `feature_importance.png` | Top 20 Random Forest feature importances |
| `seizure_onset_detection.png` | Sliding-window seizure probability over time |
| `psd_comparison.png` | Mean power spectral density: ictal vs interictal |

---

## How to Run

### Google Colab (recommended)

1. Upload `EEG_Seizure_Detection_Final.ipynb` to Google Drive
2. Open with Colab
3. Run Cell 1 (install) and Cell 2 (imports)
4. Run Cell 3 — attempts automatic download from PhysioNet
5. If download fails, follow Cell 4 instructions for manual upload
6. Run Cells 5–16 in order

**Expected runtime:** 10–15 minutes on Colab free tier (feature extraction with PAC is the slow step — ~5–8 minutes).

### Local

```bash
git clone https://github.com/Koelgeetkaur26/EEG-Based-Seizure-Detection-using-Spectral-Features
cd EEG-Based-Seizure-Detection-using-Spectral-Features
pip install -r requirements.txt
jupyter notebook EEG_Seizure_Detection_Final.ipynb
```

---

## Requirements

```
numpy
scipy
matplotlib
mne
scikit-learn
imbalanced-learn
pandas
```

Install with:
```bash
pip install mne imbalanced-learn scikit-learn scipy matplotlib numpy pandas
```

---

## Project Structure

```
EEG-Based-Seizure-Detection-using-Spectral-Features/
│
├── EEG_Seizure_Detection_Final.ipynb   # Main notebook (16 steps, fully commented)
├── README.md
├── requirements.txt
│
└── outputs/                            # Generated when notebook is run
    ├── raw_eeg_visualization.png
    ├── feature_distributions.png
    ├── model_evaluation.png
    ├── feature_importance.png
    ├── seizure_onset_detection.png
    └── psd_comparison.png
```

---

## Theoretical Background

### Why spectral features work

The power spectral density of EEG follows a characteristic 1/f shape during normal background activity — power decreasing smoothly with frequency. During a seizure, this shape is disrupted: slow hypersynchronous discharge elevates low-frequency power, and fast rhythmic bursting elevates gamma. Integrating the PSD within each of five biologically defined bands gives five numbers that compactly describe this disruption.

### Why PAC is the most discriminative feature

In normal cognition, theta-gamma coupling is associated with working memory and hippocampal-neocortical communication. During seizures, this coupling becomes pathologically strong — gamma bursts lock tightly to the theta phase across large cortical areas. The Modulation Index captures this by projecting the gamma amplitude envelope onto the complex unit circle at each theta phase angle. When there is no coupling, the projections cancel out and the mean magnitude is near zero. When coupling is present, projections cluster at a preferred angle and the mean magnitude is high. The result is a single number that cleanly separates ictal from interictal states.

### Why inter-channel correlation matters

Healthy EEG is characterised by largely independent activity across channels — different brain regions process different information simultaneously. Seizures propagate by recruiting neighbouring regions into synchronised firing. This hypersynchrony shows up as elevated pairwise correlation between channels, often preceding the amplitude increase that makes a seizure visible in a single-channel trace.

---

## Connection to Computational Neuroscience

This project bridges empirical EEG analysis and theoretical neural dynamics modelling. The spectral signatures detected here — theta-gamma coupling, high-frequency ripples, broadband power increase — are the same phenomena described in biophysical models of epileptogenesis, including models of Mossy Fibre Sprouting in the dentate gyrus and computational frameworks for seizure-generating microcircuits. This pipeline provides the empirical ground truth against which such computational models can be validated.




