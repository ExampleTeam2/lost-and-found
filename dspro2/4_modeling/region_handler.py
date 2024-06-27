from torch.utils.data import Dataset
import geopandas as gpd
import os
from shapely import wkt
import torch

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
BASE_PATH = PARENT_DIR + "/3_data_preparation/00_preparing/data/"


class RegionHandler(Dataset):
    def __init__(self, country_to_index=None):
        self.gdf = gpd.read_file(BASE_PATH + "admin_1_states_provinces.geojson", driver="GeoJSON", crs="EPSG:4326")
        # sort by name in geopandas dataframe for consistency and creating indices
        self.gdf = self.gdf.sort_values(by="region_name")
        # create a list of region names and middle points for easy access and indexing both in one list
        self.region_names = self.gdf["region_name"].tolist()
        self.region_middle_points = self.gdf["middle_point"].tolist()
        # convert the middle points to shapely Points
        self.region_middle_points_raw = [wkt.loads(point) for point in self.region_middle_points]
        self.region_middle_points = torch.tensor([(point.y, point.x) for point in self.region_middle_points_raw], dtype=torch.float64)
        self.region_middle_points_lists = [[point.y, point.x] for point in self.region_middle_points_raw]
        self.regions = list(zip(self.region_names, self.region_middle_points))
        # create a dictionary from region name to index
        self.region_to_index = {region: idx for idx, region in enumerate(self.region_names)}
        # create a dictionary from region index to middle point
        self.region_index_to_middle_point = {idx: point for idx, point in enumerate(self.region_middle_points_lists)}

        # create region index to country index if country_to_index is provided
        self.country_to_index = country_to_index
        self.region_index_to_country_index = None
        if self.country_to_index is not None:
            region_countries = self.gdf["country_name"].tolist()
            self.region_index_to_country_index = {}
            for region, country in zip(self.region_names, region_countries):
                if country in self.country_to_index:
                    region_index = self.region_to_index[region]
                    country_index = self.country_to_index[country]
                    self.region_index_to_country_index[region_index] = country_index

    def get_item(self, idx):
        region = self.regions[idx]
        return region

    def __getitem__(self, index):
        return self.get_item(index)
