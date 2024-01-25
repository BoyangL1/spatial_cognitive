import pandas as pd
import numpy as np
import math
from tqdm import tqdm

def calculate_distance(row):
    lat1, lon1 = row['phi_o'], row['lambda_o']
    lat2, lon2 = row['phi_d'], row['lambda_d']
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * \
        math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    r = 6371  # earth radius , km

    return c * r * 1000


def calculate_direction(row):
    lat1, lon1 = row['phi_o'], row['lambda_o']
    lat2, lon2 = row['phi_d'], row['lambda_d']

    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Calculate degree
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * \
        math.cos(lat2) * math.cos(dlon)
    brng = math.atan2(y, x)

    # Turn azimuth angle to degree
    brng = math.degrees(brng)

    # normalize
    brng = (brng + 360) % 360 
    brng = (brng + 90) % 360  # east = 0

    return brng


def calculate_velocity(row):

    time_duration = (row['etime']-row['stime']).total_seconds()
    
    v = row.distance/time_duration if time_duration !=0 else 0

    return v


def get_trajectory(df):
    df['stime'] = pd.to_datetime(df['stime'])
    df['etime'] = pd.to_datetime(df['etime'])

    df['distance'] = df.apply(calculate_distance, axis=1)
    df['direction'] = df.apply(calculate_direction, axis=1)

    df['velocity'] = df.apply(calculate_velocity, axis=1)
    # Min-Max Normalization
    min_velocity = 0
    max_velocity = 10
    df['velocity'] = df['velocity'].apply(lambda x: (x - min_velocity) / (max_velocity - min_velocity))

    df['direction_radians'] = df['direction'].apply(lambda x: math.radians(x))
    df['vleft'] = df.apply(lambda row: -row.velocity *
                        math.cos(row.direction_radians), axis=1)
    df['vright'] = df.apply(lambda row: row.velocity *
                        math.cos(row.direction_radians), axis=1)
    df['vup'] = df.apply(lambda row: row.velocity *
                        math.sin(row.direction_radians), axis=1)
    df['vdown'] = df.apply(lambda row: -row.velocity *
                        math.sin(row.direction_radians), axis=1)

    result_df = pd.DataFrame()
    for index, row in tqdm(df.iterrows(), total=len(df),desc='Process Trajectory'):
        # time interpolation
        time_range = pd.date_range(start=row['stime'], end=row['etime'], freq='10S')
        # create new dataframe
        new_rows = pd.DataFrame({
            'lambda_o': np.nan, 
            'phi_o': np.nan, 
            'lambda_d': row['lambda_d'],
            'phi_d': row['phi_d'],
            'start_time': time_range,
            'end_time': row['etime'],
            'vleft': row['vleft'],
            'vright': row['vright'],
            'vup': row['vup'],
            'vdown': row['vdown']
        })

        new_rows['lambda_o'] = np.linspace(row['lambda_o'], row['lambda_d'], len(time_range))
        new_rows['phi_o'] = np.linspace(row['phi_o'], row['phi_d'], len(time_range))
        
        result_df = pd.concat([result_df, new_rows], ignore_index=True)
    
    return df.grid_id_o.tolist(),df.grid_id_d.tolist(),df.lambda_o.tolist(),df.phi_o.tolist(),df.lambda_d.tolist(),df.phi_d.tolist(),result_df.lambda_o.tolist(), result_df.phi_o.tolist(), result_df.vleft.tolist(), result_df.vright.tolist(), result_df.vup.tolist(), result_df.vdown.tolist()

if __name__=="__main__":
    file_name = './one_travel_chain.csv'
    df = pd.read_csv(file_name)
    get_trajectory(df)