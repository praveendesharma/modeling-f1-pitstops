import fastf1
import pandas as pd
import os

# Set up cache
os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')

# Load session: telemetry, laps, car data, weather
session = fastf1.get_session(2023, 'Bahrain', 'R')
session.load()

# order by driver and lapnumber to calculate certain features
laps = session.laps.sort_values(['Driver', 'LapNumber']).reset_index(drop=True)

# Drop laps with missing start info
laps = laps.dropna(subset=['LapStartDate', 'LapStartTime'])

# Convert LapStartTime to timedelta
laps['LapStartTimeTD'] = pd.to_timedelta(laps['LapStartTime'].astype(str))

# Create datetime, needed to sync with weather data
laps['LapTimeDT'] = pd.to_datetime(laps['LapStartDate']) + laps['LapStartTimeTD']

# Compute TyreAge within each stint
laps['TyreAge'] = 1  

for driver in laps['Driver'].unique():
    driver_laps = laps[laps['Driver'] == driver]
    tyre_ages = []
    for stint in driver_laps['Stint'].unique():
        stint_laps = driver_laps[driver_laps['Stint'] == stint]
        age = 1
        for _ in stint_laps.iterrows():
            tyre_ages.append(age)
            age += 1
    laps.loc[laps['Driver'] == driver, 'TyreAge'] = tyre_ages

# Bucket TyreAge
laps['TyreAgeBucket'] = pd.cut(laps['TyreAge'],
                               bins=[0,10,20,30,100],
                               labels=['0-10','11-20','21-30','30+'])

# Compute LapDeltaCategory
laps['LapDelta'] = laps.groupby('Driver')['LapTime'].diff().dt.total_seconds()
laps['LapDeltaCategory'] = pd.cut(laps['LapDelta'],
                                  bins=[-float('inf'), -0.5, 0.5, float('inf')],
                                  labels=['Faster','Normal','Slower'])

# SafetyCar
laps['SafetyCar'] = laps['TrackStatus'].apply(lambda x: 1 if x=='SC' else 0)

# RacePhase
total_laps = laps['LapNumber'].max()
def phase(lap):
    if lap <= total_laps/3:
        return 'Early'
    elif lap <= 2*total_laps/3:
        return 'Middle'
    else:
        return 'Late'
laps['RacePhase'] = laps['LapNumber'].apply(phase)

# PitNextLap
laps['PitStop'] = (~laps['PitInTime'].isna()).astype(int)
laps['PitNextLap'] = laps.groupby('Driver')['PitStop'].shift(-1).fillna(0).astype(int)

# Prepare weather merge
weather = session.weather_data.copy()

# Convert weather 'Time' (timedelta) to datetime relative to session.date
weather['LapTimeDT'] = pd.to_datetime(session.date) + weather['Time']

# Sort for merge_asof
laps = laps.sort_values('LapTimeDT')
weather = weather.sort_values('LapTimeDT')

# Merge closest previous weather record per lap
laps = pd.merge_asof(laps, weather, left_on='LapTimeDT', right_on='LapTimeDT', direction='backward')

# Drop any laps that still have missing weather data
laps = laps.dropna(subset=['TrackTemp', 'Rainfall'])

# Categorize TrackTemp and Weather
def categorize_track_temp(temp):
    if temp < 30:
        return 'Low'
    elif temp < 40:
        return 'Medium'
    else:
        return 'High'

def categorize_weather(rain):
    if rain == 0:
        return 'Dry'
    elif rain < 0.5:
        return 'LightRain'
    else:
        return 'Rain'

laps['TrackTemp'] = laps['TrackTemp'].apply(categorize_track_temp)
laps['Weather'] = laps['Rainfall'].apply(categorize_weather)

# Get relevant columns and save
df = laps[['Driver','LapNumber','Compound','TyreAge','TyreAgeBucket',
           'LapDeltaCategory','TrackTemp','Weather','SafetyCar','RacePhase','PitNextLap']]

df.to_csv('bahrain_2023_lap_dataset.csv', index=False)
print("Dataset saved as 'bahrain_2023_lap_dataset.csv'")

print(df.info())
print(df.head())