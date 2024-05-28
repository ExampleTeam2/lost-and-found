import os
import geojson
import geopandas as gpd
from shapely.geometry import Point
import pyproj
import pandas as pd
import numpy as np
import re
import json
import pycountry
import concurrent
import reverse_geocoder as rg
from fuzzywuzzy import process
import sys
sys.path.insert(0, '../../')
from data_loader import resolve_env_variable, get_json_files, load_json_file
from countryconverter import convert_from_coordinates, convert_from_names
from countryconverter import convert_coordinates_to_country_names



class RegionEnricher:
    def __init__(self):
        self.file_path = '../../data/admin_1_states_provinces.json'
        self.process()

    def load_geojson(self, file):
        gdf = gpd.read_file(filename=file, driver='GeoJSON')
        return gdf
    
    # function to get the middle point of a polygon and add it to the dataframe as a new column
    def get_list_of_middle_points(self, gdf):
        gdf['middle_point'] = gdf['geometry'].centroid
        return gdf
    
    # function to get the country name from the middle point of a polygon and add it to the dataframe as a new column
    # it need a tuple of coordinates as input
    def get_country_from_middle_point(self, gdf):
        coordinates = [(point.y, point.x) for point in gdf['middle_point']]
    
        # Get country names and codes in batch
        country_names, country_codes = convert_coordinates_to_country_names(coordinates)
        
        # Assign results back to the dataframe
        gdf['country_name'] = country_names
        gdf['country_code'] = country_codes
        return gdf
    
    def process(self):
        self.gdf = self.load_geojson(self.file_path)
        # to crs 
        self.gdf = self.gdf.to_crs(epsg=4326)
        self.gdf = self.get_list_of_middle_points(self.gdf)
        # print first 5 rows of the new column
        print(self.gdf['middle_point'].head())
        self.gdf = self.get_country_from_middle_point(self.gdf)
        # print first 5 rows of the new column
        print(self.gdf['country_name'].head())
        print(self.gdf['country_code'].head())
        # to wkts
        self.gdf['middle_point'] = self.gdf['middle_point'].to_wkt()
        self.gdf.to_file('../../data/admin_1_states_provinces.geojson', driver='GeoJSON')


    def load_enriched_geojson(self):
        gdf = gpd.read_file('../../data/admin_1_states_provinces.geojson', driver='GeoJSON')
        return gdf
    
    # function to get the nearest 5 regions to a point and if the point is in a region, it will return the region names
    def get_region_from_point(self, gdf, point):
        # get the nearest 5 regions to the point
        nearest_regions = gdf['name_1'].iloc[gdf.distance(point).argsort()[:5]]
        return nearest_regions
    
    def is_point_in_region(self, gdf, point):
        # check if the point is in a region
        is_in_region = gdf.contains(point)
        return is_in_region
    
    # function to call the get_region_from_point and is_point_in_region functions for a list of points
    def get_region_from_points(self, gdf, points):
        regions = []
        is_in_regions = []
        for point in points:
            region = self.get_region_from_point(gdf, point)
            is_in_region = self.is_point_in_region(gdf, point)
            is_in_regions.append(is_in_region)
            regions.append(region)
        return regions, is_in_regions
    
    def add_regions_to_json(self, coordinates, file_map, json_files, gdf):
        # get the region names and if the point is in a region
        regions, is_in_regions = self.get_region_from_points(gdf, coordinates)
        for region, is_in_region, coord in zip(regions, is_in_regions, coordinates):
            image_ids = file_map[str(coord)]
            for image_id in image_ids:
                json_files[image_id]['region_name'] = region
                json_files[image_id]['is_in_region'] = is_in_region
        return json_files


