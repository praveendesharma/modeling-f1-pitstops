import fastf1
import pandas as pd
import os
from tqdm import tqdm
# Set up cache
os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')
output_filename = 'laps_binned_2019_2023.csv'

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


def phase(lap):
    if lap <= total_laps/3:
        return 'Early'
    elif lap <= 2*total_laps/3:
        return 'Middle'
    else:
        return 'Late'

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


all_laps = pd.DataFrame()
for year, gp, session_id in tqdm(races, desc="Processing races"):

    # Load session: telemetry, laps, car data, weather
    session = fastf1.get_session(year, gp, session_id)
    session.load()

    # order by driver and lapnumber to calculate certain features
    laps = session.laps.sort_values(['Driver', 'LapNumber']).reset_index(drop=True)


    # Drop laps with missing start info
    laps = laps.dropna(subset=['LapStartDate', 'LapStartTime', 'LapTime'])
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
                                bins=[0,5,10,15,20,30,100],
                                labels=['0-4','5-9','10-14','15-19','20-30','30+'])

    # Compute LapDeltaCategory
    laps['LapDelta'] = laps.groupby('Driver')['LapTime'].diff().dt.total_seconds()
    laps['LapDeltaCategory'] = pd.cut(laps['LapDelta'],
                                    bins=[-float('inf'), -0.5, 0.5, float('inf')],
                                    labels=['Faster','Normal','Slower'])
    laps['LapDeltaCategory'] = laps['LapDeltaCategory'].cat.add_categories(['Unknown']).fillna('Unknown')

    # SafetyCar
    # laps['SafetyCar'] = laps['TrackStatus'].apply(lambda x: 1 if x=='SC' else 0)
    laps['SafetyCar'] = laps['TrackStatus'].astype(str).str.contains('4').astype(int)
    # RacePhase
    total_laps = laps['LapNumber'].max()
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


    laps['TrackTemp'] = laps['TrackTemp'].apply(categorize_track_temp)
    laps['Weather'] = laps['Rainfall'].apply(categorize_weather)

# Get relevant columns and save
    df = laps[['Driver','LapNumber','Compound','TyreAge','TyreAgeBucket',
            'LapDeltaCategory','TrackTemp','Weather','SafetyCar','RacePhase','PitNextLap']]
    df['Year'] = year
    df['GrandPrix'] = gp
    all_laps = pd.concat([all_laps, df], ignore_index=True)
    # Save the dataframe for the current race to a separate file
    # race_filename = f"race_data_{year}_{gp.replace(' ', '_')}.csv"
    all_laps.to_csv(output_filename, index=False)
all_laps.to_csv(output_filename, index=False)

print(all_laps.info())
print(all_laps.head())
