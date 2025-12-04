import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# --- 1. LOAD DATA ---
filename = '2023_races.csv'
df = pd.read_csv(filename)

# Ensure data is sorted by Driver and Lap 
df = df.sort_values(['Year', 'GrandPrix', 'Driver', 'LapNumber']).reset_index(drop=True)

# --- 2. ENGINEER FEATURES ---
# Create a "stint progress" feature: TyreAge normalized by typical stint length per compound
# Soft ~15-20 laps, Medium ~25-30 laps, Hard ~35-40 laps
typical_stint_length = {0: 18, 1: 28, 2: 38, 3: 25, 4: 20}  # S, M, H, I, W
df['TypicalStint'] = df['Compound_Num'].map(typical_stint_length).fillna(25)
df['StintProgress'] = df['TyreAge'] / df['TypicalStint']  # >1 means overdue for pit

# Focus on strategy-relevant features only
feature_cols = [
    'TyreAge',       # Absolute tyre age
    'StintProgress', # Normalized stint progress (>1 = overdue)
    'LapDelta',      # Performance degradation signals tyre wear
]

print(f"Training on features: {feature_cols}")

# --- 3. CREATE SEQUENCES (one per driver-race) ---
X = []        
lengths = [] 
sequence_info = []  # Track which (year, gp, driver) each sequence belongs to
included_indices = []  # Track which df rows are included

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
            if len(driver_data) < 5:  # Skip very short stints
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
# State 0: Normal Racing (early/mid stint - fresh tyres)
# State 1: Pit Window (late stint - worn tyres, approaching pit)
n_states = 2

model = hmm.GaussianHMM(
    n_components=n_states,
    covariance_type="full",  # Full covariance to capture feature correlations
    n_iter=500,
    random_state=42,
    init_params='mc',  # Only init means and covariances, we set start/trans
)

# Priors based on typical race structure:
# ~70% of laps are "normal racing", ~30% are in "pit window" (last few laps of each stint)
model.startprob_ = np.array([0.9, 0.1])  # Usually start in racing state
model.transmat_ = np.array([
    [0.85, 0.15],   # From racing: can transition to pit window as tyres age
    [0.30, 0.70]    # From pit window: stay in pit window or go back to racing (after pit)
])

print("Training GaussianHMM...")
model.fit(X_scaled, lengths)

# --- 6. PREDICT HIDDEN STATES ---
print("Predicting states...")
hidden_states = model.predict(X_scaled, lengths)

# Assign hidden states only to included rows
df['HiddenState'] = -1  # Default for excluded rows
df.loc[included_indices, 'HiddenState'] = hidden_states

# Filter to only included rows for analysis
df_valid = df[df['HiddenState'] >= 0].copy()

# --- 7. IDENTIFY WHICH STATE IS "PIT WINDOW" ---
# The pit window state should have HIGHER actual pit probability
pit_probs_by_state = df_valid.groupby('HiddenState')['PitNextLap'].mean()
pit_state_idx = pit_probs_by_state.idxmax()
racing_state_idx = 1 - pit_state_idx

# --- 8. PRINT STATISTICS ---
print("\n" + "="*50)
print("           MODEL RESULTS")
print("="*50)

# State distribution
state_counts = df_valid['HiddenState'].value_counts().sort_index()
total_laps = len(df_valid)
print("\n--- State Distribution ---")
print(f"  State {racing_state_idx} (Racing):     {state_counts.get(racing_state_idx, 0):>6} laps ({100*state_counts.get(racing_state_idx, 0)/total_laps:.1f}%)")
print(f"  State {pit_state_idx} (Pit Window): {state_counts.get(pit_state_idx, 0):>6} laps ({100*state_counts.get(pit_state_idx, 0)/total_laps:.1f}%)")

# Pit probability per state
print("\n--- Pit Probability by State ---")
pit_stats = df_valid.groupby('HiddenState')['PitNextLap'].agg(['mean', 'sum', 'count'])
pit_stats.columns = ['Pit Probability', 'Actual Pits', 'Total Laps']
pit_stats.index = pit_stats.index.map(lambda x: f"State {x} ({'Pit Window' if x == pit_state_idx else 'Racing'})")
print(pit_stats.round(4))

# Learned parameters (unscaled for interpretability)
state_means_scaled = pd.DataFrame(model.means_, columns=feature_cols)
print("\n--- State Means (Scaled Features) ---")
print(state_means_scaled.round(3))

# --- 9. VISUALIZATION: ALL DRIVERS IN ONE RACE (GRID LAYOUT) ---
viz_year = 2023
viz_gp = 'Bahrain'  # First race of 2023

# Get all drivers for this race
race_df = df_valid[(df_valid['Year'] == viz_year) & (df_valid['GrandPrix'] == viz_gp)]
drivers_in_race = sorted(race_df['Driver'].unique())
n_drivers = len(drivers_in_race)

print(f"\n--- Visualizing: {viz_year} {viz_gp} GP ({n_drivers} drivers) ---")

# Grid layout: 4 rows x 5 cols = 20 drivers
n_cols = 5
n_rows = (n_drivers + n_cols - 1) // n_cols  # Ceiling division

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
    
    # Plot tyre age
    ax.plot(laps, tyre_age, 'b-', linewidth=1.5)
    
    # Highlight pit window state
    is_pit_window = (hidden == pit_state_idx)
    if len(tyre_age) > 0:
        y_max = max(tyre_age) * 1.1
        ax.fill_between(laps, 0, y_max, where=is_pit_window, color='red', alpha=0.3)
        ax.set_ylim(0, y_max)
    
    # Mark actual pit stops
    for pit_lap in actual_pits:
        ax.axvline(pit_lap, color='green', linestyle='--', linewidth=1.5)
    
    ax.set_title(driver, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Only show y-label on leftmost column
    if idx % n_cols == 0:
        ax.set_ylabel('Tyre Age', fontsize=8)
    
    # Only show x-label on bottom row
    if idx >= (n_rows - 1) * n_cols:
        ax.set_xlabel('Lap', fontsize=8)

# Hide unused subplots
for idx in range(n_drivers, len(axes1)):
    axes1[idx].set_visible(False)

# Add legend to figure
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='blue', linewidth=2, label='Tyre Age'),
    Patch(facecolor='red', alpha=0.3, label='Predicted Pit Window'),
    Line2D([0], [0], color='green', linestyle='--', linewidth=2, label='Actual Pit Stop')
]
fig1.legend(handles=legend_elements, loc='upper right', fontsize=10)
fig1.suptitle(f'Tyre Age - {viz_year} {viz_gp} GP (All Drivers)', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('gaussian_hmm_tyre_age_grid.png', dpi=150)
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
    
    # Plot lap delta
    ax.plot(laps, lap_delta, 'orange', linewidth=1.5)
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Highlight pit window state
    is_pit_window = (hidden == pit_state_idx)
    if len(lap_delta) > 0:
        y_min = min(lap_delta) - 2
        y_max = max(lap_delta) + 2
        ax.fill_between(laps, y_min, y_max, where=is_pit_window, color='red', alpha=0.3)
        ax.set_ylim(y_min, y_max)
    
    # Mark actual pit stops
    for pit_lap in actual_pits:
        ax.axvline(pit_lap, color='green', linestyle='--', linewidth=1.5)
    
    ax.set_title(driver, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Only show y-label on leftmost column
    if idx % n_cols == 0:
        ax.set_ylabel('Lap Δ (sec)', fontsize=8)
    
    # Only show x-label on bottom row
    if idx >= (n_rows - 1) * n_cols:
        ax.set_xlabel('Lap', fontsize=8)

# Hide unused subplots
for idx in range(n_drivers, len(axes2)):
    axes2[idx].set_visible(False)

# Add legend to figure
legend_elements2 = [
    Line2D([0], [0], color='orange', linewidth=2, label='Lap Delta'),
    Patch(facecolor='red', alpha=0.3, label='Predicted Pit Window'),
    Line2D([0], [0], color='green', linestyle='--', linewidth=2, label='Actual Pit Stop')
]
fig2.legend(handles=legend_elements2, loc='upper right', fontsize=10)
fig2.suptitle(f'Lap Delta - {viz_year} {viz_gp} GP (All Drivers)', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('gaussian_hmm_lap_delta_grid.png', dpi=150)
plt.show()

print("\nVisualizations saved:")
print("  - gaussian_hmm_tyre_age_grid.png")
print("  - gaussian_hmm_lap_delta_grid.png")
