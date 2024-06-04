import sys
import os
working_directory = os.path.abspath('.')
sys.path.append(working_directory)

import numpy as np
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd

import json
from collections import namedtuple
import pickle

# todo: location of home and workplace
Traveler = namedtuple('Traveler', ['who', 'migrt', 'visit_date'])

def add_fnid(df):
    df['geometry_o'] = [Point(xy) for xy in zip(df['lambda_o'], df['phi_o'])]
    df['geometry_d'] = [Point(xy) for xy in zip(df['lambda_d'], df['phi_d'])]
    grid_gdf = gpd.read_file("./data/shenzhen_grid/shenzhen_grid.shp")
    geo_df_o = df.copy()
    geo_df_o = gpd.GeoDataFrame(geo_df_o, geometry='geometry_o')
    geo_df_d = df.copy()
    geo_df_d = gpd.GeoDataFrame(geo_df_d, geometry='geometry_d')

    # Ensure CRS matches before joining
    geo_df_o.set_crs(grid_gdf.crs, inplace=True)
    geo_df_d.set_crs(grid_gdf.crs, inplace=True)

    result_o = gpd.sjoin(geo_df_o, grid_gdf, how="left", predicate="within")
    result_d = gpd.sjoin(geo_df_d, grid_gdf, how="left", predicate="within")

    # add columns for the fnid
    df['grid_id_o'] = result_o['fnid']  
    df['grid_id_d'] = result_d['fnid']
    return df

def join_fnid(df, grid_gdf):
    # 在这里加一个ss_city_grid就成了，不用空间计算。
    df['geometry_o'] = [Point(xy) for xy in zip(df['lambda_o'], df['phi_o'])]
    df['geometry_d'] = [Point(xy) for xy in zip(df['lambda_d'], df['phi_d'])]
    df = pd.merge(df, grid_gdf, how='left', left_on=['org_chess_x', 'org_chess_y'], right_on=['CHESS_X', 'CHESS_Y'])
    df.rename(columns={'FNID': 'grid_id_o'}, inplace=True)    
    df = pd.merge(df, grid_gdf, how='left', left_on=['dst_chess_x', 'dst_chess_y'], right_on=['CHESS_X', 'CHESS_Y'])
    df.rename(columns={'FNID': 'grid_id_d'}, inplace=True)
    df.drop(columns=['CHESS_X_x', 'CHESS_Y_x', 'geometry_x', 'CHESS_X_y', 'CHESS_Y_y', 'geometry_y'], inplace=True)
    return df


def skip_empty_row(df):
# Finding rows with at least one NaN value, out of city
    columns_to_check = ["lambda_o","phi_o","lambda_d","phi_d","grid_id_o","grid_id_d"]
    rows_with_nan = df[columns_to_check].isna().any(axis=1)
    nan_indexes = df[rows_with_nan].index
    
    nan_indexes_stack = nan_indexes.tolist()
    # for i in nan_indexes[::2]:
    #     df.at[i, 'etime'] = df.at[i + 1, 'etime']
    #     df.loc[i] = df.loc[i].combine_first(df.loc[i + 1])
    # df.drop(nan_indexes[1::2], inplace=True)

    while nan_indexes_stack:
        i = nan_indexes_stack.pop(0)
        j = nan_indexes_stack[0]
        k = -3 # any negative integer number expect -1 and 0
        if i + 1 == j:
            df.loc[i] = df.loc[i].combine_first(df.loc[j])
            k = nan_indexes_stack.pop(0)
            df.drop(index=j, inplace=True)
        elif k + 1 == i:
            df.loc[k] = df.loc[k].combine_first(df.loc[i])
            df.drop(index=i, inplace=True)
        else:
            df.drop(index=i, inplace=True)
            
    df.fillna(0, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def check_discountinuous(df):
    discontinuous_rows = []
    for i in range(len(df) - 1):
        if df['lambda_d'].iloc[i] != df['lambda_o'].iloc[i + 1]:
            # next row index
            discontinuous_rows.append(i + 1)
    return discontinuous_rows

def remove_discontinuous(df):
    discontinuous_rows = check_discountinuous(df)
    if len(discontinuous_rows) > 0:
        for idx in discontinuous_rows:
            df.loc[idx, 'poi_o'] = df.loc[idx - 1, 'poi_d']
            df.loc[idx, 'org_chess_x'] = df.loc[idx - 1, 'dst_chess_x']
            df.loc[idx, 'org_chess_y'] = df.loc[idx - 1, 'dst_chess_y']
            df.loc[idx, 'lambda_o'] = df.loc[idx - 1, 'lambda_d']
            df.loc[idx, 'phi_o'] = df.loc[idx - 1, 'phi_d']
            df.loc[idx, 'pre_chess_x'] = df.loc[idx - 1, 'post_chess_x']
            df.loc[idx, 'pre_chess_y'] = df.loc[idx - 1, 'post_chess_y']
            df.loc[idx, 'grid_id_o'] = df.loc[idx-1, 'grid_id_d']

    discontinuous_rows_after = check_discountinuous(df)
    if len(discontinuous_rows_after) > 0:
        print('Discontinuous rows found: {}'.format(discontinuous_rows_after))
        raise ValueError('Error: discontinuous rows found.')
    else: return df

def drop_unnecessary_columns(df):
    columns_to_drop = [
        'poi_o','poi_d','org_chess_x', 'org_chess_y', 'dst_chess_x',
        'dst_chess_y', 'pre_chess_x', 'pre_chess_y', 'post_chess_x', 'post_chess_y',
        'geometry_o', 'geometry_d','home_distance','trip_distance'
    ]
    df = df.drop(columns=columns_to_drop)
    return df

def create_coords_mapping(df, trg_path):
    df['origin'] = df.apply(lambda row: (row['lambda_o'], row['phi_o']), axis=1)
    df['destination'] = df.apply(lambda row: (row['lambda_d'], row['phi_d']), axis=1)
    unique_coords_ids = pd.unique(df[['origin', 'destination']].values.ravel('K'))
    # using factorize function to apply new unique id
    # new_codes 是数字编码
    # unique 是唯一坐标序列
    new_codes, unique = pd.factorize(unique_coords_ids, sort=True)
    coords_id_mapping = dict(zip(unique, new_codes))
    df['origin_id'] = df['origin'].map(coords_id_mapping)
    df['destination_id'] = df['destination'].map(coords_id_mapping)

    # 这里的grid_id_o/d都是fnid
    origin_to_fnid = df.groupby('origin')['grid_id_o'].first().to_dict()
    destination_to_fnid = df.groupby('destination')['grid_id_d'].first().to_dict()
    coords_fnid_mapping = {**origin_to_fnid, **destination_to_fnid}

    with open(trg_path + "/coords_fnid_mapping.pkl", "wb") as f:
        pickle.dump(coords_fnid_mapping, f)
    id_coords_mapping = dict(zip(new_codes,unique))
    with open(trg_path + "/id_coords_mapping.pkl", "wb") as f:
        pickle.dump(id_coords_mapping, f)
    return df

def build_chain(group):
    chains = []
    current_chain = [group['origin'].iloc[0]]
    
    for i in range(len(group) - 1):
        current_chain.append(group['destination'].iloc[i])
        # Check if the next 'o' is different from the current 'd'
        if group['destination'].iloc[i] != group['origin'].iloc[i + 1]:
            chains.append(current_chain)
            current_chain = [group['origin'].iloc[i + 1]]
    
    # Append the last destination and the final chain
    current_chain.append(group['destination'].iloc[-1])
    chains.append(current_chain)
    
    return chains

def build_id_chain(group):
    chains = []
    current_chain = [group['origin_id'].iloc[0]]
    
    for i in range(len(group) - 1):
        current_chain.append(group['destination_id'].iloc[i])
        # Check if the next 'o' is different from the current 'd'
        if group['destination_id'].iloc[i] != group['origin_id'].iloc[i + 1]:
            chains.append(current_chain)
            current_chain = [group['origin_id'].iloc[i + 1]]
    
    # Append the last destination and the final chain
    current_chain.append(group['destination_id'].iloc[-1])
    chains.append(current_chain)
    
    return chains

def build_fnid_chain(group):
    chains = []
    current_chain = [group['grid_id_o'].iloc[0]]

    for i in range(len(group) - 1):
        current_chain.append(group['grid_id_d'].iloc[i])
        # Check if the next 'o' is different from the current 'd'
        if group['grid_id_d'].iloc[i] != group['grid_id_o'].iloc[i + 1]:
            chains.append(current_chain)
            current_chain = [group['grid_id_o'].iloc[i + 1]]
    
    # Append the last destination and the final chain
    current_chain.append(group['grid_id_d'].iloc[-1])
    chains.append(current_chain)
    
    return chains

def build_both_chains(group):
    travel_chain = build_chain(group)
    id_chain = build_id_chain(group)
    fnid_chain = build_fnid_chain(group)
    return pd.Series({'travel_chain': travel_chain, 'id_chain': id_chain, 'fnid_chain':fnid_chain})

def create_json_travel_chains(df, trg_path):
    # Group by 'date' and apply the chain-building function
    grouped1 = df[df['date']<df['migrt']]
    grouped2 = df[df['date']>df['migrt']]
    grouped1 = grouped1.groupby('date').apply(build_both_chains).reset_index()
    grouped2 = grouped2.groupby('date').apply(build_both_chains).reset_index()
    # Define the namedtuple type
    TravelData = namedtuple('TravelChain', ['date', 'travel_chain','id_chain','fnid_chain'])
    # Convert each row to a namedtuple
    namedtuples_list1 = [TravelData(row.date, row.travel_chain[0], row.id_chain[0], row.fnid_chain[0]) for _, row in grouped1.iterrows()]
    namedtuples_list2 = [TravelData(row.date, row.travel_chain[0], row.id_chain[0], row.fnid_chain[0]) for _, row in grouped2.iterrows()]
    namedtuples_all = namedtuples_list1+namedtuples_list2
    
    def int64_converter(obj):
        if isinstance(obj, np.int64):
            return int(obj)
        raise TypeError

    dicts_list1 = [nt._asdict() for nt in namedtuples_list1]
    with open(trg_path + '/before_migrt.json', 'w') as file:
        json.dump(dicts_list1, file, indent=4,default=int64_converter)

    dicts_list2 = [nt._asdict() for nt in namedtuples_list2]
    with open(trg_path + '/after_migrt.json', 'w') as file:
        json.dump(dicts_list2, file, indent=4,default=int64_converter)

    dicts_list_all = [nt._asdict() for nt in namedtuples_all]
    with open(trg_path + '/all_traj.json', 'w') as file:
        json.dump(dicts_list_all, file, indent=4,default=int64_converter)

def mergeFeatureDataframes(df):

    selectedColumns1 = ['grid_id_d', 'pre_home_distance', 'post_home_distance', 'LU_Business',
                        'LU_City_Road', 'LU_Consumption', 'LU_Culture', 'LU_Industry',
                        'LU_Medical', 'LU_Park_&_Scenery', 'LU_Public', 'LU_Residence',
                        'LU_Science_&_Education', 'LU_Special', 'LU_Transportation', 'LU_Wild'
                        ]
    
    featureDf1 = df[selectedColumns1].drop_duplicates(subset=['grid_id_d'])
    featureDf1 = featureDf1.rename(columns={'grid_id_d':'fnid'})

    selectedColumns2 = ['grid_id_o', 'LU_Business',
                        'LU_City_Road', 'LU_Consumption', 'LU_Culture', 'LU_Industry',
                        'LU_Medical', 'LU_Park_&_Scenery', 'LU_Public', 'LU_Residence',
                        'LU_Science_&_Education', 'LU_Special', 'LU_Transportation', 'LU_Wild'
                        ]

    featureDf2 = df[selectedColumns2].drop_duplicates(subset=['grid_id_o'])
    featureDf2 = featureDf2.rename(columns={'grid_id_o':'fnid'})

    mergedDf = featureDf1.merge(featureDf2, on='fnid', how='outer')
    mergedDf = mergedDf.fillna(0)

    for col in selectedColumns2:
        if col != 'fnid' and col in featureDf1.columns and col in featureDf2.columns:
            mergedDf[col] = mergedDf[col + '_x'] + mergedDf[col + '_y']
            mergedDf.drop([col + '_x', col + '_y'], axis=1, inplace=True)

    mergedDf.fillna(0, inplace=True)
    return mergedDf

def split_feature_data(df, trg_path):
    df_before_migrt = df[df['date'] < df['migrt']]
    df_after_migrt = df[df['date']>df['migrt']]

    before_feature=mergeFeatureDataframes(df_before_migrt)
    before_feature.to_csv(trg_path + '/before_migrt_feature.csv',index=False)

    after_feature = mergeFeatureDataframes(df_after_migrt)
    after_feature.to_csv(trg_path + '/after_migrt_feature.csv',index=False)

    all_feature = mergeFeatureDataframes(df)
    all_feature.to_csv(trg_path + '/all_traj_feature.csv',index=False)


if __name__ == '__main__':
    user_chain_src_path = './data/user_chains/'
    filenames = [f for f in os.listdir(user_chain_src_path) if f.endswith('.csv')]
    extract_who = lambda s: int(s.rstrip('.csv').split('_')[-1])
    who_list = list(map(extract_who, filenames))

    user_chain_trg_path = './data/user_data/'
        
    grid_gdf = gpd.read_file("./data/shenzhen_grid/grid/ss_city_grid_by_cover/ss_city_grid_by_cover.shp")

    # create new directories:
    trg_paths = []
    for who in who_list:
        one_trg_path = user_chain_trg_path + '{:08d}'.format(who)
        os.makedirs(one_trg_path, exist_ok=True)
        print(f'Creating directory {one_trg_path}')
        trg_paths.append(one_trg_path)

    for i, filename in enumerate(filenames):
        df = pd.read_csv(user_chain_src_path + filename)
        # df = add_fnid(df)
        df = join_fnid(df, grid_gdf)
        df = skip_empty_row(df)
        df = remove_discontinuous(df)
        df = drop_unnecessary_columns(df)
        split_feature_data(df, trg_path=trg_paths[i])
        df = create_coords_mapping(df, trg_path=trg_paths[i])
        create_json_travel_chains(df, trg_path=trg_paths[i])

        traveler = Traveler(who=who_list[i], migrt=df['migrt'].iloc[0], visit_date=df['date'].unique().tolist())
        with open(trg_paths[i] + '/traveler_info.pkl', 'wb') as f:
            pickle.dump(traveler, f)