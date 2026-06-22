# Modeling Formula 1 Pit Stop Decisions

Probabilistic models for predicting when Formula 1 drivers are likely to pit, built from lap-level race data. This repository contains the code and datasets behind our technical report on modeling pit-stop strategy using Bayesian networks and hidden Markov models (HMMs).

## Technical Report

The full write-up is available on Google Drive:

**[F1 Pit Stop Modeling — Technical Report](https://drive.google.com/file/d/1Ry2pIaU3t3ZJfLLGLk2NFx7gUlBj-QPu/view?usp=sharing)**

## Overview

Pit-stop timing is a central strategic decision in F1. We frame it as a sequence modeling problem: each lap produces observations (tyre age, pace, compound, weather, safety-car status, race phase), and the model infers whether the driver is in a **racing** state or a **pit window** state.

Three approaches are implemented:

| Model | Script | Description |
|-------|--------|-------------|
| **Static Bayesian Network** | `static_bn.py` | Discrete BN with maximum-likelihood parameter estimation; models causal links from tyre compound, pace, weather, and safety car to the pit decision on the next lap. |
| **Discrete HMM** | `discrete_hmm.py` | Categorical HMM over binned lap features (tyre age bucket, lap delta, race phase, safety car); trained on multi-season race data. |
| **Gaussian HMM** | `gaussian_hmm.py` | Continuous HMM over tyre age, stint progress, and lap delta; produces per-driver visualizations of predicted pit windows vs. actual stops. |

## Data

Lap-level features are extracted from [FastF1](https://github.com/theOehrly/Fast-F1) for races from 2019–2023 (2020 omitted). Key fields include:

- **TyreAge** / **TyreAgeBucket** — laps on current compound
- **LapDeltaCategory** — pace relative to previous lap (Faster / Normal / Slower)
- **Compound** — tyre compound (Soft, Medium, Hard, etc.)
- **SafetyCar** — safety-car period indicator
- **Weather** / **TrackTemp** — conditions at lap time
- **RacePhase** — Early / Middle / Late
- **PitNextLap** — target label (1 if driver pits on the following lap)

### Included datasets

| File | Description |
|------|-------------|
| `laps_binned_2019_2023.csv` | Multi-season binned lap data (2019–2023) for discrete HMM |
| `laps_continuous_2019_2023.csv` | Continuous-feature variant of the full dataset |
| `laps_continuous_2023.csv` | 2023 season laps for Gaussian HMM |
| `bahrain_2023_laps_binned.csv` | Bahrain 2023 subset for Bayesian network |
| `bahrain_2023_laps_continuous.csv` | Continuous Bahrain 2023 data |
| `bahrain_2023_discrete_hmm_predictions.csv` | Discrete HMM predictions on Bahrain 2023 |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

FastF1 caches downloaded session data in `cache/` (also gitignored).

## Usage

Run scripts from the project root. Pre-built CSVs are included, so you can run the models directly without re-fetching data.

### 1. Extract / refresh data (optional)

```bash
python data_exploration.py        # binned features → laps_binned_2019_2023.csv
python data_exploration_cont.py   # continuous features → laps_continuous_2019_2023.csv
```

These scripts pull race sessions via FastF1 and may take a while on first run.

### 2. Train models

```bash
python static_bn.py       # Bayesian network on Bahrain 2023
python discrete_hmm.py    # categorical HMM on multi-season data
python gaussian_hmm.py    # Gaussian HMM on 2023 data + plots
```

`gaussian_hmm.py` saves visualization PNGs (`bahrain_2023_gaussian_hmm_*.png`) comparing predicted pit windows against actual pit stops.

## Project structure

```
modeling-f1-pitstops/
├── data_exploration.py          # FastF1 → binned lap dataset
├── data_exploration_cont.py     # FastF1 → continuous lap dataset
├── static_bn.py                 # Static Bayesian network
├── discrete_hmm.py              # Discrete (categorical) HMM
├── gaussian_hmm.py              # Gaussian HMM + visualizations
├── requirements.txt
├── laps_*.csv                   # Processed lap-level datasets
├── bahrain_2023_*.csv           # Bahrain 2023 subsets and predictions
└── bahrain_2023_gaussian_hmm_*.png  # Model output figures
```

## Dependencies

- [FastF1](https://github.com/theOehrly/Fast-F1) — F1 timing and telemetry API
- [pgmpy](https://pgmpy.org/) — Bayesian networks
- [hmmlearn](https://hmmlearn.readthedocs.io/) — Hidden Markov models
- pandas, numpy, scikit-learn, matplotlib, seaborn, tqdm

## License

Code and technical report for academic use. F1 timing data is subject to [FastF1](https://github.com/theOehrly/Fast-F1) and Ergast/API terms.
