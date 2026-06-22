import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# --- 1. LOAD DATA ---
filename = 'laps_continuous_2023.csv'
df = pd.read_csv(filename)
df = df.sort_values(['Year', 'GrandPrix', 'Driver', 'LapNumber']).reset_index(drop=True)

# --- 2. ENGINEER FEATURES ---
typical_stint_length = {0: 18, 1: 28, 2: 38, 3: 25, 4: 20}
df['TypicalStint'] = df['Compound_Num'].map(typical_stint_length).fillna(25)
df['StintProgress'] = df['TyreAge'] / df['TypicalStint']

feature_cols = ['TyreAge', 'StintProgress', 'LapDelta']
print(f"Training on features: {feature_cols}")

# --- 3. CREATE SEQUENCES ---
X = []        
lengths = [] 
sequence_info = []
included_indices = []

years = df['Year'].unique()
sessions = df['GrandPrix'].unique()

for year in years:
    for session in sessions:
        session_data = df[(df['Year'] == year) & (df['GrandPrix'] == session)]
        if session_data.empty:
            continue
        session_drivers = session_data['Driver'].unique()
        
        for driver in session_drivers:
            driver_mask = (df['Year'] == year) & (df['GrandPrix'] == session) & (df['Driver'] == driver)
            driver_data = df[driver_mask]
            if len(driver_data) < 5:
                continue
            obs = driver_data[feature_cols].values
            X.append(obs)
            lengths.append(len(obs))
            sequence_info.append((year, session, driver))
            included_indices.extend(driver_data.index.tolist())

X_concat = np.concatenate(X)

# --- 4. SCALE FEATURES ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_concat)

# --- 5. TRAIN GAUSSIAN HMM ---
n_states = 2

model = hmm.GaussianHMM(
    n_components=n_states,
    covariance_type="full",
    n_iter=500,
    random_state=42,
    init_params='mc',
)

model.startprob_ = np.array([0.9, 0.1])
model.transmat_ = np.array([
    [0.85, 0.15],
    [0.30, 0.70]
])

print("Training GaussianHMM...")
model.fit(X_scaled, lengths)

# --- 6. PREDICT HIDDEN STATES ---
print("Predicting states...")
hidden_states = model.predict(X_scaled, lengths)

df['HiddenState'] = -1
df.loc[included_indices, 'HiddenState'] = hidden_states
df_valid = df[df['HiddenState'] >= 0].copy()

# --- 7. IDENTIFY PIT WINDOW STATE ---
pit_probs_by_state = df_valid.groupby('HiddenState')['PitNextLap'].mean()
pit_state_idx = pit_probs_by_state.idxmax()
racing_state_idx = 1 - pit_state_idx

# --- 8. PRINT STATISTICS ---
print("\n" + "="*50)
print("           MODEL RESULTS")
print("="*50)

state_counts = df_valid['HiddenState'].value_counts().sort_index()
total_laps = len(df_valid)
print("\n--- State Distribution ---")
print(f"  State {racing_state_idx} (Racing):     {state_counts.get(racing_state_idx, 0):>6} laps ({100*state_counts.get(racing_state_idx, 0)/total_laps:.1f}%)")
print(f"  State {pit_state_idx} (Pit Window): {state_counts.get(pit_state_idx, 0):>6} laps ({100*state_counts.get(pit_state_idx, 0)/total_laps:.1f}%)")

print("\n--- Pit Probability by State ---")
pit_stats = df_valid.groupby('HiddenState')['PitNextLap'].agg(['mean', 'sum', 'count'])
pit_stats.columns = ['Pit Probability', 'Actual Pits', 'Total Laps']
pit_stats.index = pit_stats.index.map(lambda x: f"State {x} ({'Pit Window' if x == pit_state_idx else 'Racing'})")
print(pit_stats.round(4))

state_means_scaled = pd.DataFrame(model.means_, columns=feature_cols)
print("\n--- State Means (Scaled Features) ---")
print(state_means_scaled.round(3))

# --- 9. VISUALIZATION ---
viz_year = 2023
viz_gp = 'Bahrain'

race_df = df_valid[(df_valid['Year'] == viz_year) & (df_valid['GrandPrix'] == viz_gp)]
drivers_in_race = sorted(race_df['Driver'].unique())
n_drivers = len(drivers_in_race)

print(f"\n--- Visualizing: {viz_year} {viz_gp} GP ({n_drivers} drivers) ---")

n_cols = 5
n_rows = (n_drivers + n_cols - 1) // n_cols

# ============== FIGURE 1: TYRE AGE GRID ==============
fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(20, 12), sharex=True)
axes1 = axes1.flatten()

for idx, driver in enumerate(drivers_in_race):
    ax = axes1[idx]
    driver_data = race_df[race_df['Driver'] == driver].sort_values('LapNumber')
    
    if driver_data.empty:
        ax.set_visible(False)
        continue
    
    laps = driver_data['LapNumber'].values
    tyre_age = driver_data['TyreAge'].values
    hidden = driver_data['HiddenState'].values
    actual_pits = driver_data[driver_data['PitNextLap'] == 1]['LapNumber'].values
    
    ax.plot(laps, tyre_age, 'b-', linewidth=1.5)
    
    is_pit_window = (hidden == pit_state_idx)
    if len(tyre_age) > 0:
        y_max = max(tyre_age) * 1.1
        ax.fill_between(laps, 0, y_max, where=is_pit_window, color='red', alpha=0.3)
        ax.set_ylim(0, y_max)
    
    for pit_lap in actual_pits:
        ax.axvline(pit_lap, color='green', linestyle='--', linewidth=1.5)
    
    ax.set_title(driver, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if idx % n_cols == 0:
        ax.set_ylabel('Tyre Age', fontsize=8)
    if idx >= (n_rows - 1) * n_cols:
        ax.set_xlabel('Lap', fontsize=8)

for idx in range(n_drivers, len(axes1)):
    axes1[idx].set_visible(False)

legend_elements = [
    Line2D([0], [0], color='blue', linewidth=2, label='Tyre Age'),
    Patch(facecolor='red', alpha=0.3, label='Predicted Pit Window'),
    Line2D([0], [0], color='green', linestyle='--', linewidth=2, label='Actual Pit Stop')
]
fig1.legend(handles=legend_elements, loc='upper right', fontsize=10)
fig1.suptitle(f'Tyre Age - {viz_year} {viz_gp} GP (All Drivers)', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('bahrain_2023_gaussian_hmm_tyre_age.png', dpi=150)
plt.show()

# ============== FIGURE 2: LAP DELTA GRID ==============
fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(20, 12), sharex=True)
axes2 = axes2.flatten()

for idx, driver in enumerate(drivers_in_race):
    ax = axes2[idx]
    driver_data = race_df[race_df['Driver'] == driver].sort_values('LapNumber')
    
    if driver_data.empty:
        ax.set_visible(False)
        continue
    
    laps = driver_data['LapNumber'].values
    lap_delta = driver_data['LapDelta'].values
    hidden = driver_data['HiddenState'].values
    actual_pits = driver_data[driver_data['PitNextLap'] == 1]['LapNumber'].values
    
    ax.plot(laps, lap_delta, 'orange', linewidth=1.5)
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    is_pit_window = (hidden == pit_state_idx)
    if len(lap_delta) > 0:
        y_min = min(lap_delta) - 2
        y_max = max(lap_delta) + 2
        ax.fill_between(laps, y_min, y_max, where=is_pit_window, color='red', alpha=0.3)
        ax.set_ylim(y_min, y_max)
    
    for pit_lap in actual_pits:
        ax.axvline(pit_lap, color='green', linestyle='--', linewidth=1.5)
    
    ax.set_title(driver, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if idx % n_cols == 0:
        ax.set_ylabel('Lap Δ (sec)', fontsize=8)
    if idx >= (n_rows - 1) * n_cols:
        ax.set_xlabel('Lap', fontsize=8)

for idx in range(n_drivers, len(axes2)):
    axes2[idx].set_visible(False)

legend_elements2 = [
    Line2D([0], [0], color='orange', linewidth=2, label='Lap Delta'),
    Patch(facecolor='red', alpha=0.3, label='Predicted Pit Window'),
    Line2D([0], [0], color='green', linestyle='--', linewidth=2, label='Actual Pit Stop')
]
fig2.legend(handles=legend_elements2, loc='upper right', fontsize=10)
fig2.suptitle(f'Lap Delta - {viz_year} {viz_gp} GP (All Drivers)', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('bahrain_2023_gaussian_hmm_lap_delta.png', dpi=150)
plt.show()

print("\nVisualizations saved:")
print("  - bahrain_2023_gaussian_hmm_tyre_age.png")
print("  - bahrain_2023_gaussian_hmm_lap_delta.png")
