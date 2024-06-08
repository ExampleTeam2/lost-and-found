from torch.utils.data import Dataset
import geopandas as gpd
import os
import fiona
from shapely import wkt
import torch

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

class RegionHandler(Dataset):
    def __init__(self):
        print(fiona.supported_drivers)
        self.gdf = gpd.read_file('./../data/admin_1_states_provinces.geojson', driver='GeoJSON', crs='EPSG:4326')
        # create a sorted list of region names and middle points for easy access and indexing both in one list
        self.region_names = self.gdf['region_name'].tolist()
        self.region_middle_points = self.gdf['middle_point'].tolist()
        # convert the middle points to shapely Points
        self.region_middle_points = [wkt.loads(point) for point in self.region_middle_points]
        self.region_middle_points = torch.tensor([(point.x, point.y) for point in self.region_middle_points], dtype=torch.float64)
        # create a list of tuples with region name and middle point sorted by region name
        self.regions = sorted(list(zip(self.region_names, self.region_middle_points)), key=lambda x: x[0])

    def get_item(self, idx):
        region = self.regions[idx]
        return region
    
    def __getitem__(self, index):
        return self.get_item(index)
    
    def get_idx(self, region_name):
        return self.region_names.index(region_name)
    
    def get_middle_point_by_list_of_idx(self, idx_list):
        return [self.region_middle_points[idx] for idx in idx_list]
