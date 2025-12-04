import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder

def build_observation_string(row):
    """
    Build a discrete observation string using the strongest predictors.
    Removed features that dilute the HMM signal.
    """
    features = [
        row["TyreAgeBucket"],      # e.g. New / Med / Old / VOld
        row["LapDeltaCategory"],   # e.g. Fast / Norm / Slow (your categories)
        row["RacePhase"],          # Early / Middle / Late
        row["SafetyCar"],       # NoSC / VSC / SC
    ]
    return "-".join(features)


# --- 1. LOAD DATA ---
df = pd.read_csv('all_race_data_binned.csv')
df = df.sort_values(['Driver', 'LapNumber']).reset_index(drop=True)

# --- 2. PREPARE DISCRETE FEATURES ---
mappings = {
    'TyreAgeBucket': {'0-4':'VNew','5-9': 'New', '10-14': 'Med', '15-19':'Med-old','20-30': 'Old', '30+': 'VOld'},
    'LapDeltaCategory': {'Faster': 'Fast', 'Normal': 'Norm', 'Unknown': 'Norm', 'Slower': 'Slow'},
    'Compound': {'SOFT': 'S', 'MEDIUM': 'M', 'HARD': 'H', 'INTERMEDIATE': 'I', 'WET': 'W'},
    'SafetyCar': {0: 'NoSC', 1: 'SC'},
    'Weather': {'Dry': 'Dry', 'LightRain': 'Damp', 'Rain': 'Wet'},
    'TrackTemp': {'Low': 'Cold', 'Medium': 'Warm', 'High': 'Hot'}
    }

# Apply mappings
for col, mapping in mappings.items():
    if col in df.columns:
        df[col] = df[col].map(mapping).fillna('Unknown')

# Define the full list of categorical features to include in the observation
feature_cols = [
    'TyreAgeBucket', 
    'LapDeltaCategory', 
    'Compound', 
    'SafetyCar',
    'Weather', 
    'TrackTemp', 
    'RacePhase'
]


# --- 3. COMBINE INTO SINGLE OBSERVATION ---
# Basically like "Old-Slow-S-NoSC-Dry-Warm-End"

# df['Observation_Str'] = df[feature_cols].astype(str).agg('-'.join, axis=1)
df["Observation_Str"] = df.apply(build_observation_string, axis=1)


print("\nSample combined observations:")
print(df['Observation_Str'].head())

# Encode into integers 
encoder = LabelEncoder()
df['Observation_Int'] = encoder.fit_transform(df['Observation_Str'])

# Calculate Vocabulary Size 
n_features = len(encoder.classes_)
print(f"\nTotal unique observation types (Vocabulary Size): {n_features}")

# --- 4. PREPARE SEQUENCES ---
X = []
lengths = []

drivers = df['Driver'].unique()
sessions = df['GrandPrix'].unique()
years = df['Year'].unique()
for year in years:
    for session in sessions:
        session_data = df[(df['Year'] == year) & (df['GrandPrix'] == session)]
        session_drivers = session_data['Driver'].unique()
        
        for driver in session_drivers:
            driver_data = session_data[session_data['Driver'] == driver]
            driver_obs = driver_data[['Observation_Int']].values
            X.append(driver_obs)
            lengths.append(len(driver_obs))
            
X_concat = np.concatenate(X)

# --- 5. TRAIN CATEGORICAL HMM ---
# n_components=2: State 0 (Racing/Stay Out) vs State 1 (Pit Window/In-Lap)
"""
model = hmm.CategoricalHMM(n_components=2,
                        n_features=n_features, 
                        random_state=42,
                        n_iter=2000,
                        transmat_prior = np.array([
                                [0.95, 0.05],
                                [0.95, 0.05]
                                ]),
                        startprob_prior=np.array([0.95, 0.05]),
                        emissionprob_prior = np.random.dirichlet(np.ones(n_features), size=2),
                        init_params=''
                        ) 


model.transmat_ = np.array([
    [0.95, 0.05],
    [0.95, 0.05]
])

model.startprob_ = np.array([0.95, 0.05])

print(model.startprob_prior.shape)
"""
model = hmm.CategoricalHMM(
    n_components=2,
    n_features=n_features,
    random_state=42,
    n_iter=1000,
    init_params='ste',   # allow startprob, transmat, emission to be learned
) 


# model.emissionprob_ = np.random.dirichlet(np.ones(n_features), size=2)
alpha = np.full(n_features, 50.0)
model.emissionprob_ = np.random.dirichlet(alpha, size=2)

print("\nTraining CategoricalHMM...")
model.fit(X_concat, lengths)

# --- 6. PREDICT & ANALYZE ---
print("Predicting Hidden States...")
hidden_states = model.predict(X_concat, lengths)
df['HiddenState'] = hidden_states

# --- VALIDATION ---
print("\n" + "="*40)
print("       MODEL RESULTS")
print("="*40)

# 1. Pit Probability per State
# We identify the 'Pit Window' state as the one with higher PitNextLap probability
pit_probs = df.groupby('HiddenState')['PitNextLap'].mean()
print("\n[Pit Probability per State]")
print(pit_probs)

pit_state = pit_probs.idxmax()
racing_state = 1 - pit_state
print(f"\n> State {pit_state} identified as 'PIT WINDOW' (Higher probability of pitting next lap)")

# 2. Top Observations in the Pit State
# What combination of factors drives the model to predict a Pit Stop?
print(f"\n[Most Frequent Observations in 'Pit Window' State (State {pit_state})]")
pit_state_data = df[df['HiddenState'] == pit_state]
top_obs = pit_state_data['Observation_Str'].value_counts().head(5)
print(top_obs)

print(f"\n[Most Frequent Observations in 'Racing' State (State {racing_state})]")
race_state_data = df[df['HiddenState'] == racing_state]
print(race_state_data['Observation_Str'].value_counts().head(5))

# --- FIXED VISUALIZATION CODE FOR ONE DRIVER ---

import matplotlib.pyplot as plt

sample_driver = drivers[0]
sample_year   = df['Year'].min()
sample_gp     = df['GrandPrix'].unique()[0]

d_slice = df[
    (df['Driver'] == sample_driver) &
    (df['Year'] == sample_year) &
    (df['GrandPrix'] == sample_gp)
].copy()

fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Tyre Age if available
if 'TyreAge' in d_slice.columns:
    ax1.set_xlabel('Lap Number')
    ax1.set_ylabel('Tyre Age', color='tab:blue')
    ax1.plot(d_slice['LapNumber'], d_slice['TyreAge'], color='tab:blue')
else:
    ax1.set_xlabel('Lap Number')
    ax1.set_ylabel('State')

# Highlight the pit window state
is_pit_state = (d_slice['HiddenState'] == pit_state).astype(int)
ylims = ax1.get_ylim()

ax1.fill_between(
    d_slice['LapNumber'],
    ylims[0], ylims[1],
    where=is_pit_state == 1,
    color='red', alpha=0.25,
    label=f'Predicted Pit Window (State {pit_state})'
)

# ---------------------------
# CORRECT PIT DETECTION
# ---------------------------
pit_stops = d_slice.index[d_slice["TyreAge"].diff() < 0]

# Keep only the lap where tyre age resets (TyreAge == 1)
pit_stops = [i for i in pit_stops if d_slice.loc[i, "TyreAge"] == 1]

# Draw vertical lines **only once**
for pit in pit_stops:
    ax1.axvline(d_slice.loc[pit, "LapNumber"], color='black', linestyle='--', linewidth=2)

# Remove duplicate legend entries
handles, labels = ax1.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.title(f"Discrete HMM Strategy Detection — Driver {sample_driver}")
plt.tight_layout()
plt.show()


# 3. Save Results
df.to_csv('bahrain_2023_discrete_predictions.csv', index=False)
print("\nPredictions saved to 'bahrain_2023_discrete_predictions.csv'")