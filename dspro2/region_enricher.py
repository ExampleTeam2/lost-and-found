import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import reverse_geocoder as rg
from fuzzywuzzy import process
from countryconverter import convert_coordinates_to_country_names, convert_country_from_names
import warnings
from scipy.spatial import cKDTree
import os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

BASE_PATH = CURRENT_DIR + "/3_data_preparation/00_preparing/data/"


class RegionEnricher:
    def __init__(self):
        self.geojson_file_path = BASE_PATH + "admin_1_states_provinces.geojson"
        self.file_path = BASE_PATH + "admin_1_states_provinces.json"
        warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS")

    def load_geojson(self, file):
        gdf = gpd.read_file(filename=file, driver="GeoJSON")
        return gdf

    # function to get the middle point of a polygon and add it to the dataframe as a new column
    def get_list_of_middle_points(self, gdf):
        gdf["middle_point"] = gdf["geometry"].centroid
        gdf["inscribed_point"] = gdf["geometry"].representative_point()
        return gdf

    # function to get the country name from the middle point of a polygon and add it to the dataframe as a new column
    # it need a tuple of coordinates as input
    def get_country_from_middle_point(self, gdf):
        coordinates = [(point.y, point.x) for point in gdf["inscribed_point"]]

        # Get country names and codes in batch
        country_names, country_codes = convert_coordinates_to_country_names(coordinates)

        # Assign results back to the dataframe
        gdf["country_name_old"] = country_names
        gdf["country_code_old"] = country_codes
        # get a country name from the field admin and process with convert_country_from_names
        country_names, country_codes = convert_country_from_names(gdf["admin"].values)
        gdf["country_name"] = country_names
        gdf["country_code"] = country_codes
        # if country name is missing, use the old one
        gdf["country_name"] = gdf["country_name"].fillna(gdf["country_name_old"])
        gdf["country_code"] = gdf["country_code"].fillna(gdf["country_code_old"])
        all_countries = gdf["country_name"].unique()
        all_countries_old = gdf["country_name_old"].unique()
        print(set(all_countries) - set(all_countries_old))
        print(set(all_countries_old) - set(all_countries))
        gdf["region_name"] = gdf["country_name"] + "_" + gdf["name"] + "_" + gdf["adm1_code"]
        return gdf

    def process(self):
        self.gdf = self.load_geojson(self.file_path)
        # assert there are not missing values in the adm1_code column
        assert self.gdf["adm1_code"].notnull().all()
        # to crs
        self.gdf = self.gdf.to_crs(epsg=4326)
        self.gdf = self.get_list_of_middle_points(self.gdf)
        # print first 5 rows of the new column
        print(self.gdf["middle_point"].head())
        self.gdf = self.get_country_from_middle_point(self.gdf)
        # print first 5 rows of the new column
        print(self.gdf["country_name"].head())
        print(self.gdf["country_code"].head())
        print(self.gdf["region_name"].head())
        region_names = self.gdf["region_name"].values
        # assert that there are no missing values for region names
        assert self.gdf["region_name"].notnull().all()
        print(region_names[:5])
        print(region_names[-5:])
        print(len(region_names))
        print(len(set(region_names)))
        print(self.gdf["region_name"].value_counts()[:40])
        # assert that the region names are unique
        assert len(region_names) == len(set(region_names))
        # to wkts
        self.gdf["middle_point"] = self.gdf["middle_point"].to_wkt()
        self.gdf["inscribed_point"] = self.gdf["inscribed_point"].to_wkt()
        self.gdf.to_file(self.geojson_file_path, driver="GeoJSON", encoding="utf-8")

    def load_enriched_geojson(self):
        gdf = gpd.read_file(self.geojson_file_path, driver="GeoJSON", crs="EPSG:4326")
        region_names = gdf["region_name"].values
        # assert that the region names are unique and there are no missing values
        assert gdf["region_name"].notnull().all()
        assert len(region_names) == len(set(region_names))
        return gdf

    def get_region_from_point(self, gdf, point):
        if not isinstance(point, Point):
            point = Point(point)
        # get the nearest 5 regions to the point
        nearest_regions = gdf["region_name"].iloc[gdf.distance(point).argsort()[:5]]
        return nearest_regions

    def is_point_in_region(self, gdf, point):
        if not isinstance(point, Point):
            point = Point(point)
        # check if the point is in a region
        is_in_region = gdf.contains(point).any()
        return is_in_region

    # Find the top 5 nearest regions for each point
    def find_nearest_regions(self, points, regions, k=5):
        # Get the centroids of the regions
        centroids = np.array(list(regions.geometry.centroid.apply(lambda x: (x.x, x.y))))

        # Create a KDTree for fast nearest-neighbor lookup
        tree = cKDTree(centroids)

        # Initialize results
        results = [[[regions.iloc[idx]["region_name"], regions.iloc[idx]["middle_point"]] for idx in tree.query((point.y, point.x), k=k)[1]] for point in points]

        return results

    def get_region_from_points(self, gdf, points):
        gdf = gdf.to_crs(epsg=4326)

        # Convert points to a GeoSeries
        points_gs = gpd.GeoSeries([Point(point) if not isinstance(point, Point) else point for point in points], crs="EPSG:4326")

        # Check if the points are in any of the regions
        is_in_regions = [gdf.contains(point).any() for point in points_gs]
        print("Finished checking if points are in regions.  Now finding nearest regions...")

        nearest_regions = self.find_nearest_regions(points_gs, gdf)
        print("Finished finding nearest regions.")

        return nearest_regions, is_in_regions

    def add_regions_to_json(self, coordinates, file_map, json_files, gdf):
        regions, is_in_regions = self.get_region_from_points(gdf, coordinates)
        for region, is_in_region, coord in zip(regions, is_in_regions, coordinates):
            image_ids = file_map[str(coord)]
            for image_id in image_ids:
                json_files[image_id]["regions"] = region
                json_files[image_id]["is_in_region"] = str(is_in_region)
        return json_files
