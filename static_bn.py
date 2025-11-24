from pgmpy.models import DiscreteBayesianNetwork
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator

df = pd.read_csv("bahrain_2023_lap_dataset.csv")

# assuring data types
for col in ['TyreAgeBucket','Compound','LapDeltaCategory','TrackTemp','Weather','RacePhase']:
    df[col] = df[col].astype(str)
df['SafetyCar'] = df['SafetyCar'].astype(int)
df['PitNextLap'] = df['PitNextLap'].astype(int)


model = DiscreteBayesianNetwork([
    ('Compound', 'TyreAgeBucket'),
    ('TyreAgeBucket', 'LapDeltaCategory'),
    ('LapDeltaCategory', 'PitNextLap'),

    ('Weather', 'TrackTemp'),
    ('TrackTemp', 'PitNextLap'),

    ('SafetyCar', 'PitNextLap'),
    ('RacePhase', 'PitNextLap')
])

model.fit(df, estimator=MaximumLikelihoodEstimator)

for cpd in model.get_cpds():
    print(cpd)