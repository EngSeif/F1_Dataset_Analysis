import pandas as pd
import datetime as dt
from xlsxwriter import Workbook

# File path
fpath = './Dataset/'

try:
    circuits = pd.read_csv(f'{fpath}circuits.csv', index_col=0, na_values=r'\N')
    drivers = pd.read_csv(f'{fpath}drivers.csv', index_col=0, na_values=r'\N')
    pitStops = pd.read_csv(f'{fpath}pit_stops.csv')
    results = pd.read_csv(f'{fpath}results.csv', index_col=0, na_values=r'\N')
    qualifying = pd.read_csv(f'{fpath}qualifying.csv', index_col=0, na_values=r'\N')
    lapTimes = pd.read_csv(f'{fpath}lap_times.csv')
    status = pd.read_csv(f'{fpath}status.csv', index_col=0, na_values=r'\N')
    races = pd.read_csv(f'{fpath}races.csv', na_values=r'\N')
except FileNotFoundError as e:
    print(f"Error loading file: {e}")
    raise

# Data formatting
drivers['driverName'] = drivers['forename'] + ' ' + drivers['surname']
races['date'] = pd.to_datetime(races['date'])
pitStops['seconds'] = pitStops['milliseconds'] / 1000
results['seconds'] = results['milliseconds'] / 1000

# Core merge: Pit stops with race results
merged = pd.merge(
    pitStops,
    results[['raceId', 'driverId', 'position', 'points', 'statusId']],
    on=['raceId', 'driverId'],
    how='left'
)

# Add qualifying positions
merged = pd.merge(
    merged,
    qualifying[['raceId', 'driverId', 'position']].rename(columns={'position': 'qualifyingPos'}),
    on=['raceId', 'driverId'],
    how='left'
)

# Add average lap times
lap_avg = lapTimes.groupby(['raceId', 'driverId'])['milliseconds'].mean().reset_index()
merged = pd.merge(merged, lap_avg.rename(columns={'milliseconds': 'avgLapTime'}), how='left')

# Add driver names and race details
merged = pd.merge(merged, drivers[['driverName']], left_on='driverId', right_index=True)
merged = pd.merge(merged, races[['raceId', 'year', 'name', 'circuitId']].rename(columns={'name': 'raceName'}))

# Add status information
merged = pd.merge(merged, status[['status']], left_on='statusId', right_index=True)

# Save to Excel (using default engine)
merged.to_excel('pit_stop_analysis_full.xlsx', index=False)

print("Successfully created pit_stop_analysis_full.xlsx")
