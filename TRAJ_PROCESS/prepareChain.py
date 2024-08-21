import sys
import os
working_directory = os.path.abspath('.')
sys.path.append(working_directory)

import numpy as np
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd

import json
import pickle
from collections import namedtuple
from SCBIRL_Global_PE.utils import Traveler, training_baseline_count

def build_chain(group):
    lon = group.longitude.tolist()
    lat = group.latitude.tolist()
    coords = list(zip(lon, lat))
    return coords


def build_id_chain(group):
    id_list = group.mapid.tolist()    
    return id_list


def build_fnid_chain(group):
    fnid_list = group.fnid.tolist()    
    return fnid_list


def build_both_chains(group):
    travel_chain = build_chain(group)
    id_chain = build_id_chain(group)
    fnid_chain = build_fnid_chain(group)
    return pd.Series({'travel_chain': travel_chain, 'id_chain': id_chain, 'fnid_chain':fnid_chain})


def int64_converter(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    raise TypeError


def featureset2Json(data):
    # Group by 'date' and apply the chain-building function
    grouped = data.groupby('date').apply(build_both_chains).reset_index()
    # Define the namedtuple type
    TravelData = namedtuple('TravelChain', ['date', 'travel_chain','id_chain','fnid_chain'])
    # Convert each row to a namedtuple
    namedtuples_all = [TravelData(row.date, row.travel_chain, row.id_chain, row.fnid_chain) for _, row in grouped.iterrows()]
    
    dicts_list_all = [nt._asdict() for nt in namedtuples_all]
    return dicts_list_all


def featureset2allFeature(data):
    columns = data.columns[data.columns.str.startswith("LU_")].tolist()
    columns.insert(0, 'fnid')
    res = data[columns].drop_duplicates(subset = 'fnid')
    return res.reset_index(drop = True)


def createCoordsMapping(data):
    coords_series = data.apply(lambda row: (row['longitude'], row['latitude']), axis=1)
    # concatenate the two Series
    # get unique coords
    unique_coords_ids = coords_series.drop_duplicates()
    # using factorize function to apply new unique id
    # new_codes 是数字编码, unique 是唯一坐标序列
    new_codes, unique = pd.factorize(unique_coords_ids, sort=True)
    id_coords_mapping = dict(zip(new_codes, unique))
    coords_id_mapping = {coords: id for id, coords in id_coords_mapping.items()}
    # compute the id
    data['mapid'] = coords_series.map(coords_id_mapping)
    fnid_base_id = data.groupby('mapid')['fnid'].first().tolist()
    coords_base_id = [id_coords_mapping[name] for name, _ in data.groupby('mapid')]
    # create the mapping
    coords_fnid_mapping = dict(zip(coords_base_id, fnid_base_id))

    return coords_fnid_mapping, id_coords_mapping


def writing2DataFolder(data):
    who = data.who.tolist()[0]
    output_path = './data/user_data/'
    writing_path = output_path + f'{who:09d}/'
    
    os.makedirs(writing_path, exist_ok=True)
    
    all_feature = featureset2allFeature(data)
    coords_fnid_mapping, id_coords_mapping = createCoordsMapping(data)
    all_json = featureset2Json(data)
    visited_date = data['date'].unique().tolist()
    if len(visited_date) < 3 * training_baseline_count:
        iter_start_date = visited_date[np.ceil(len(visited_date)/3).astype(int) - 1]
    else:
        iter_start_date = visited_date[training_baseline_count - 1]
    traveler = Traveler(who=who, visit_date=visited_date, iter_start_date=iter_start_date)

    all_feature.to_csv(writing_path + 'all_traj_feature.csv',index=False)
    
    with open(writing_path + 'all_traj.json', 'w') as file:
        json.dump(all_json, file, indent=4, default=int64_converter)    
    with open(writing_path + "coords_fnid_mapping.pkl", "wb") as f:
        pickle.dump(coords_fnid_mapping, f)
    with open(writing_path + "id_coords_mapping.pkl", "wb") as f:
        pickle.dump(id_coords_mapping, f)
    with open(writing_path + 'traveler_info.pkl', 'wb') as f:
        pickle.dump(traveler, f)



if __name__ == '__main__':
    read_path = './data/'
    
    df = pd.read_csv(read_path + 'featureset.csv')
    df_lists = [df for name, df in df.groupby('who')]
    
    for i in range(len(df_lists)):
        data = df_lists[i]
        writing2DataFolder(data)
    
    