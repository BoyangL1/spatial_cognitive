import geopandas as gpd
import pickle
import numpy as np
import math
from scipy.spatial import cKDTree
from tqdm import tqdm
import real_trajeccotry as TRAJ
import periodic_cann as UP
from joblib import Parallel, delayed
import pandas as pd

class GridModel:
    def __init__(self):
        print("Initialize grid model")
        self.n = 2**7  # number of neurons
        self.dt = 0.5
        self.tau = 5  # neuron time constant

        # Envelope and Weight Matrix Parameters
        self.wmag = 2.4
        self.wtphase = 2
        self.alpha = 1

        # Envelope and Weight Matrix parameters
        self.x = np.arange(-self.n/2, self.n/2)
        self.envelope_all = True
        if not self.envelope_all:
            self.a_envelope = np.exp(-4 * (np.outer(self.x**2, np.ones(self.n)) + np.outer(np.ones(self.n), self.x**2)) / (self.n/2)**2)
        else:
            self.a_envelope = 0.6 * np.ones((self.n, self.n))

        # initialize weight matrix, input envelope and r mask
        self.l_inhibition = [8,10,12,14,16]
        self.h_grid = len(self.l_inhibition) # layers of grid cells
        self.w, self.w_u, self.w_d, self.w_l, self.w_r, self.r_l_mask, self.r_r_mask, self.r_u_mask, self.r_d_mask = UP.initializeWeight(self.n, self.h_grid, self.l_inhibition, self.wmag, self.wtphase)

        # initialize grid pattern
        self.r = np.zeros((self.h_grid, self.n, self.n))

def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    radius_of_earth = 6371  
    distance = radius_of_earth * c * 1000  
    return distance


def calculate_direction(lon1,lat1,lon2,lat2):
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

    # Normalize to 0-360 degrees
    brng = (brng + 360) % 360

    return brng

def intervalCoords(distance_meters,anchor_coords, fnid_coords, v=10, interval=10):
    steps = int(math.ceil(distance_meters / (v * interval)))
    lon_step = (fnid_coords[0] - anchor_coords[0]) / steps
    lat_step = (fnid_coords[1] - anchor_coords[1]) / steps
    x = [anchor_coords[0]]
    y = [anchor_coords[1]]
    for step in range(1, steps):
        new_lon = anchor_coords[0] + lon_step * step
        new_lat = anchor_coords[1] + lat_step * step
        x.append(new_lon)
        y.append(new_lat)
    x.append(fnid_coords[0])
    y.append(fnid_coords[1])
    return x,y

def updateCoordsGrid(anchor_grid_code,GM, anchor_coords, fnid_coords):

    r = anchor_grid_code

    distance_meters = haversine_distance(anchor_coords[1], anchor_coords[0], fnid_coords[1], fnid_coords[0])
    direction_degrees = calculate_direction(anchor_coords[0], anchor_coords[1],fnid_coords[0], fnid_coords[1])

    # Simulate Trajectory Data
    v = 10 # 10m/s
    max_v = 40 # 40m/s
    interval = 10 # s
    anchor_list = [fnid_coords]
    steps = int(math.ceil(distance_meters / (v * interval))) + 1

    v = v/max_v
    vleft = [-v*math.sin(math.radians(direction_degrees))]*steps
    vright = [v*math.sin(math.radians(direction_degrees))]*steps
    vup = [v*math.cos(math.radians(direction_degrees))]*steps
    vdown = [-v*math.cos(math.radians(direction_degrees))]*steps
    
    [x,y] = intervalCoords(distance_meters,anchor_coords, fnid_coords, v, interval)
    [_,r,coords_grid_dic]=UP.runCann(GM.dt, GM.tau, GM.n, anchor_list, x,y, vleft, vright, vup, vdown, GM.a_envelope, GM.alpha, r, GM.r_r_mask, GM.r_l_mask, GM.r_u_mask, GM.r_d_mask, GM.w_r, GM.w_l, GM.w_d, GM.w_u, [GM.n//2,GM.n//2])
    
    return coords_grid_dic

if __name__=="__main__":
    sz_gdf = gpd.read_file('./data/shenzhen_grid/nanshan_grid.shp')
    sz_gdf['lon'] = sz_gdf['geometry'].centroid.x
    sz_gdf['lat'] = sz_gdf['geometry'].centroid.y

    coords_file_path = './data/coords_grid_data.pkl'
    with open(coords_file_path, 'rb') as file:
        coords_grid_data = pickle.load(file)

    anchor_points = list(coords_grid_data.keys())
    kdtree = cKDTree(anchor_points) # create a k-d tree for search o(nlogn)

    nearest_coords = []
    for index, row in sz_gdf.iterrows():
        lon = row['lon']
        lat = row['lat']
        _, idx = kdtree.query((lon, lat))
        nearest_coord = anchor_points[idx]
        nearest_coords.append(nearest_coord)
    sz_gdf['nearest_coords'] = nearest_coords

    GM = GridModel()
    def process_row(row):
        try:
            anchor_coords = row['nearest_coords']
            fnid_coords = (row['lon'], row['lat'])
            anchor_grid_code = coords_grid_data.get(anchor_coords)
            return updateCoordsGrid(anchor_grid_code, GM, anchor_coords, fnid_coords)
        except Exception as e:
            print(f"Error processing row: {e}")
        return None

    fnid_grid_dic = {}
    results = Parallel(n_jobs=-1)(delayed(process_row)(row) for index, row in tqdm(sz_gdf.iterrows(), total=sz_gdf.shape[0]))

    for result in filter(None, results):
        fnid_grid_dic.update(result)

    with open('./data/nanshan_grid_all.pkl', 'wb') as file:
        pickle.dump(fnid_grid_dic, file)