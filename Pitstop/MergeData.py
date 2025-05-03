import pandas as pd

# Load your main dataset
df = pd.read_excel('pit_stop_analysis_full.xlsx')
# Load constructor and results data
constructors = pd.read_csv('./Dataset/constructors.csv', index_col=0, na_values=r'\N')
results = pd.read_csv('./Dataset/results.csv', index_col=0, na_values=r'\N')

# Prepare constructor mapping
constructor_map = results[['raceId', 'driverId', 'constructorId']].drop_duplicates()
constructors = constructors.rename(columns={'name': 'constructorName', 'nationality': 'constructorNationality'})

# Merge constructor ID into main dataframe
df = pd.merge(df, constructor_map, on=['raceId', 'driverId'], how='left')

# Merge constructor details
df = pd.merge(df, constructors[['constructorName', 'constructorNationality']], left_on='constructorId', right_index=True, how='left')

# Load or create weather data
try:
    weather = pd.read_csv('./Dataset/F1 Weather(2023-2018).csv')
except FileNotFoundError:
    # Dummy weather data (if real data is not available)
    import numpy as np
    races = pd.read_csv('./Dataset/races.csv', na_values=r'\N')
    weather = races[['raceId']].copy()
    weather['temperature'] = np.random.normal(22, 5, size=len(weather))
    weather['conditions'] = np.random.choice(['Sunny', 'Cloudy', 'Rainy', 'Wet', 'Dry'], size=len(weather))

# Load lap times data
lapTimes = pd.read_csv('./Dataset/lap_times.csv')

# Get position before pit stop
df = pd.merge(df, lapTimes[['raceId', 'driverId', 'lap', 'position']].rename(columns={'position': 'position_before_pit'}), on=['raceId', 'driverId', 'lap'], how='left')

df.to_excel('merged_f1_data.xlsx', index=False)
print("Merged dataset saved as 'merged_f1_data.xlsx'")


