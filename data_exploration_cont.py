import fastf1
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
# 1. Setup and Load
os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')

races = [

    # -------------------------
    # 2018
    # -------------------------
    # (2018, 'Bahrain', 'R'),
    # (2018, 'China', 'R'),
    # (2018, 'Azerbaijan', 'R'),
    # (2018, 'Spain', 'R'),
    # (2018, 'Monaco', 'R'),
    # (2018, 'Canada', 'R'),
    # (2018, 'France', 'R'),
    # (2018, 'Austria', 'R'),
    # (2018, 'Great Britain', 'R'),
    # (2018, 'Germany', 'R'),
    # (2018, 'Hungary', 'R'),
    # (2018, 'Belgium', 'R'),
    # (2018, 'Italy', 'R'),
    # (2018, 'Singapore', 'R'),
    # (2018, 'Russia', 'R'),
    # (2018, 'Japan', 'R'),
    # (2018, 'United States', 'R'),
    # (2018, 'Mexico', 'R'),
    # (2018, 'Brazil', 'R'),
    # (2018, 'Abu Dhabi', 'R'),

    # -------------------------
    # 2019
    # -------------------------
    (2019, 'Australia', 'R'),
    (2019, 'Bahrain', 'R'),
    (2019, 'China', 'R'),
    (2019, 'Azerbaijan', 'R'),
    (2019, 'Spain', 'R'),
    (2019, 'Monaco', 'R'),
    (2019, 'Canada', 'R'),
    (2019, 'France', 'R'),
    (2019, 'Austria', 'R'),
    (2019, 'Great Britain', 'R'),
    (2019, 'Germany', 'R'),
    (2019, 'Hungary', 'R'),
    (2019, 'Belgium', 'R'),
    (2019, 'Italy', 'R'),
    (2019, 'Singapore', 'R'),
    (2019, 'Russia', 'R'),
    (2019, 'Japan', 'R'),
    (2019, 'Mexico', 'R'),
    (2019, 'United States', 'R'),
    (2019, 'Brazil', 'R'),
    (2019, 'Abu Dhabi', 'R'),

    # -------------------------
    # 2020 skipped
    # -------------------------

    # -------------------------
    # 2021
    # -------------------------
    (2021, 'Bahrain', 'R'),
    (2021, 'Italy', 'R'),  # Emilia Romagna
    (2021, 'Portugal', 'R'),
    (2021, 'Spain', 'R'),
    (2021, 'Monaco', 'R'),
    (2021, 'Azerbaijan', 'R'),
    (2021, 'France', 'R'),
    (2021, 'Austria', 'R'),
    (2021, 'Great Britain', 'R'),
    (2021, 'Hungary', 'R'),
    (2021, 'Belgium', 'R'),
    (2021, 'Netherlands', 'R'),
    (2021, 'Italy', 'R'),   # Italian GP
    (2021, 'Russia', 'R'),
    (2021, 'Turkey', 'R'),
    (2021, 'United States', 'R'),
    (2021, 'Mexico', 'R'),
    (2021, 'Qatar', 'R'),
    (2021, 'Saudi Arabia', 'R'),
    (2021, 'Abu Dhabi', 'R'),

    # -------------------------
    # 2022
    # -------------------------
    (2022, 'Bahrain', 'R'),
    (2022, 'Saudi Arabia', 'R'),
    (2022, 'Australia', 'R'),
    (2022, 'Italy', 'R'),  # Emilia Romagna
    (2022, 'United States', 'R'),  # Miami
    (2022, 'Spain', 'R'),
    (2022, 'Monaco', 'R'),
    (2022, 'Azerbaijan', 'R'),
    (2022, 'Canada', 'R'),
    (2022, 'Great Britain', 'R'),
    (2022, 'Austria', 'R'),
    (2022, 'France', 'R'),
    (2022, 'Hungary', 'R'),
    (2022, 'Belgium', 'R'),
    (2022, 'Netherlands', 'R'),
    (2022, 'Italy', 'R'),
    (2022, 'Singapore', 'R'),
    (2022, 'Japan', 'R'),
    (2022, 'United States', 'R'),
    (2022, 'Mexico', 'R'),
    (2022, 'Brazil', 'R'),
    (2022, 'Abu Dhabi', 'R'),

    # -------------------------
    # 2023
    # -------------------------
    (2023, 'Bahrain', 'R'),
    (2023, 'Saudi Arabia', 'R'),
    (2023, 'Australia', 'R'),
    (2023, 'Azerbaijan', 'R'),
    (2023, 'Miami', 'R'),
    (2023, 'Spain', 'R'),
    (2023, 'Monaco', 'R'),
    (2023, 'Canada', 'R'),
    (2023, 'Austria', 'R'),
    (2023, 'Great Britain', 'R'),
    (2023, 'Hungary', 'R'),
    (2023, 'Belgium', 'R'),
    (2023, 'Netherlands', 'R'),
    (2023, 'Italy', 'R'),
    (2023, 'Singapore', 'R'),
    (2023, 'Japan', 'R'),
    (2023, 'Qatar', 'R'),
    (2023, 'United States', 'R'),
    (2023, 'Mexico', 'R'),
    (2023, 'Brazil', 'R'),
    (2023, 'Las Vegas', 'R'),
    (2023, 'Abu Dhabi', 'R'),
]
output_filename = 'all_race_data_cont.csv'
i=0
all_laps = pd.DataFrame()
for year, gp, session_id in tqdm(races, desc="Processing races"):
    i+=1
    if i<=54:
        continue  

    session = fastf1.get_session(year, gp, session_id)
    session.load()

    # 2. Base Lap Data
    laps = session.laps.sort_values(['Driver', 'LapNumber']).reset_index(drop=True)

    # Drop invalid laps
    laps = laps.dropna(subset=['LapStartDate', 'LapStartTime', 'LapTime'])

    # 3. Feature: TyreAge (Continuous Integer)
    # We calculate the cumulative age of the tyre in laps
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

    # 4. Feature: LapDelta (Continuous Float)
    # Change in lap time vs previous lap (positive = slowing down)
    laps['LapDelta'] = laps.groupby('Driver')['LapTime'].diff().dt.total_seconds().fillna(0)

    # 5. Feature: Compound (Mapped to Integer)
    compound_map = {
        'SOFT': 0, 
        'MEDIUM': 1, 
        'HARD': 2, 
        'INTERMEDIATE': 3, 
        'WET': 4
    }
    laps['Compound_Num'] = laps['Compound'].map(compound_map).fillna(1).astype(int)

    # 6. Feature: SafetyCar (Binary Integer)
    laps['SafetyCar'] = laps['TrackStatus'].astype(str).str.contains('4').astype(int)

    # 7. Target: PitNextLap
    laps['PitStop'] = (~laps['PitInTime'].isna()).astype(int)
    laps['PitNextLap'] = laps.groupby('Driver')['PitStop'].shift(-1).fillna(0).astype(int)

    # 8. Weather Integration (Continuous Floats)
    weather = session.weather_data.copy()

    # Sync timestamps
    laps['LapStartTimeTD'] = pd.to_timedelta(laps['LapStartTime'].astype(str))
    laps['LapTimeDT'] = pd.to_datetime(laps['LapStartDate']) + laps['LapStartTimeTD']

    weather['LapTimeDT'] = pd.to_datetime(session.date) + weather['Time']

    laps = laps.sort_values('LapTimeDT')
    weather = weather.sort_values('LapTimeDT')

    laps = pd.merge_asof(laps, weather, left_on='LapTimeDT', right_on='LapTimeDT', direction='backward')

    # Fill missing weather with previous valid value or default
    laps['TrackTemp'] = laps['TrackTemp'].fillna(method='ffill')
    laps['Rainfall'] = laps['Rainfall'].fillna(0).astype(int) 

    # 9. Select Columns for GaussianHMM
    cols_to_keep = [
        'Driver', 
        'LapNumber', 
        'TyreAge',
        'LapDelta',        
        'Compound_Num',    
        'TrackTemp',       
        'Rainfall',        
        'SafetyCar',       
        'PitNextLap'       
    ]

    df_continuous = laps[cols_to_keep]

    df_continuous['Year'] = year
    df_continuous['GrandPrix'] = gp

    all_laps = pd.concat([all_laps, df_continuous], ignore_index=True)
    all_laps.to_csv(output_filename, index=False)  
# Save
all_laps.to_csv(output_filename, index=False)
# df_continuous.to_csv(output_filename, index=False)
print(all_laps.head())
print(all_laps.info())
