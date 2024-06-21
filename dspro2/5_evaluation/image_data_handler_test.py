import sys
import json
import requests
import io
import os
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '../4_modeling')
from custom_image_dataset import CustomImageDataset

class TestImageDataHandler:
    def __init__(self, test_path='./test_data.pth', country_to_index_path='./country_to_index.json', region_to_index_path='./region_to_index.json', region_index_to_middle_point_path='./region_index_to_middle_point.json', region_index_to_country_index_path='./region_index_to_country_index.json', batch_size=100):
        # Load the test data
        print(f"Loading test data from {os.path.basename(test_path)}")
        dataset_response = requests.get(test_path)
        dataset_response.raise_for_status()
        dataset_file = io.BytesIO(dataset_response.content)
        test_data = torch.load(dataset_file)
        print("Test data loaded.")
        if 'test_regions' in test_data:
            images, countries, coordinates, regions = test_data['test_images'], test_data['test_countries'], test_data['test_coordinates'], test_data['test_regions']
        else:
            images, countries, coordinates = test_data['test_images'], test_data['test_countries'], test_data['test_coordinates']
            # Make fake empty tensor (-1) of same length for regions
            regions = torch.full((len(images), 1), -1, dtype=torch.int64)
            
        self.num_regions = 0
        self.num_countries = 0
            
        self.country_to_index = None
        self.region_to_index = None
        self.region_index_to_middle_point = None
        self.region_index_to_country_index = None
        if country_to_index_path is not None:
          # Load the country_to_index mapping
          json_response = requests.get(country_to_index_path)
          json_response.raise_for_status()  # Check if the download was successful

          # Load the file into a variable
          self.country_to_index = json.loads(json_response.text)
          
          self.num_countries = len(self.country_to_index)
        
        if region_to_index_path is not None:
          # Load the region_to_index mapping
          json_response = requests.get(region_to_index_path)
          json_response.raise_for_status()
          
          # Load the file into a variable
          self.region_to_index = json.loads(json_response.text)
          
          self.num_regions = len(self.region_to_index)
          
        if region_index_to_middle_point_path is not None:
          # Load the region_index_to_middle_point mapping
          json_response = requests.get(region_index_to_middle_point_path)
          json_response.raise_for_status()
          
          # Load the file into a variable
          self.region_index_to_middle_point = json.loads(json_response.text)
          
        if region_index_to_country_index_path is not None:
          # Load the region_index_to_country_index mapping
          json_response = requests.get(region_index_to_country_index_path)
          json_response.raise_for_status()
          
          # Load the file into a variable
          self.region_index_to_country_index = json.loads(json_response.text)
        
        self.test_loader = DataLoader(CustomImageDataset(images, coordinates, countries, regions, country_to_index=self.country_to_index, region_to_index=self.region_to_index), batch_size=batch_size, shuffle=False)
