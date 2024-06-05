from torch.utils.data import Dataset
import geopandas as gpd

class RegionHandler(Dataset):
    def __init__(self):
        self.gdf = gpd.read_file('../../data/admin_1_states_provinces.geojson', driver='GeoJSON', crs='EPSG:4326')

        # create a sorted list of region names and middle points for easy access and indexing both in one list
        self.region_names = self.gdf['name_en'].tolist()
        self.region_middle_points = self.gdf['middle_point'].tolist()
        # create a list of tuples with region name and middle point sorted by region name
        self.regions = sorted(list(zip(self.region_names, self.region_middle_points)), key=lambda x: x[0])

    def get_item(self, idx):
        region = self.regions[idx]
        return region
    
    def get_idx(self, region_name):
        return self.region_names.index(region_name)
    
    def get_middle_point_by_list_of_idx(self, idx_list):
        return [self.region_middle_points[idx] for idx in idx_list]
