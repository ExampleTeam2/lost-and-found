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
    def __init__(self, test_path='./test_data.pth', country_to_index_path='./country_to_index.json', region_to_index_path='./region_to_index.json', region_index_to_middle_point_path='./region_index_to_middle_point.json', region_index_to_country_index_path='./region_index_to_country_index.json', batch_size=100, cache=True):
        # Load the test data
        print(f"Loading test data from {os.path.basename(test_path)}")
        
        # if test_path is a URL, get the path of the file
        test_file_identifier = None
        if test_path.startswith('http'):
          parts = test_path.split('/')
          if len(parts) > 3:
            parts = parts[3:]
            if 'files' in parts and len(parts) > 1:
              file_name = parts[-1]
              if parts[-2] == 'files':
                # remove 'files' from the path
                parts = parts[:-1]
              run_name = parts[-2]
              test_file_identifier = f"{run_name}/{file_name}"
              
        # create test file identifier path if it is not None and does not exist
        if test_file_identifier is not None:
          os.makedirs(os.path.dirname(test_file_identifier), exist_ok=True)
          
        if test_file_identifier is None or not os.path.exists(test_file_identifier):
          dataset_response = requests.get(test_path)
          dataset_response.raise_for_status()
          dataset_file = io.BytesIO(dataset_response.content)
          test_data = torch.load(dataset_file)
          print("Test data loaded.")
          # Cache the test data
          if cache and test_file_identifier is not None:
            print(f"Caching test data to {test_file_identifier}")
            with open(test_file_identifier, 'wb') as f:
              torch.save(test_data, f)
            print("Test data cached.")
        elif test_file_identifier is not None and os.path.exists(test_file_identifier):
          test_data = torch.load(test_file_identifier)
          print("Test data loaded from cache.")
          
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
          
          print(f"Loaded {len(self.country_to_index)} countries.")
          
          self.num_countries = len(self.country_to_index)
        
        if region_to_index_path is not None:
          # Load the region_to_index mapping
          json_response = requests.get(region_to_index_path)
          json_response.raise_for_status()
          
          # Load the file into a variable
          self.region_to_index = json.loads(json_response.text)
          
          print(f"Loaded {len(self.region_to_index)} regions.")
          
          self.num_regions = len(self.region_to_index)
          
        if region_index_to_middle_point_path is not None:
          # Load the region_index_to_middle_point mapping
          json_response = requests.get(region_index_to_middle_point_path)
          json_response.raise_for_status()
          
          print(f"Loaded {len(json.loads(json_response.text))} region middle points.")
          
          # Load the file into a variable
          self.region_index_to_middle_point = json.loads(json_response.text)
          # Convert the keys to integers
          self.region_index_to_middle_point = {int(k): v for k, v in self.region_index_to_middle_point.items()}
          
        if region_index_to_country_index_path is not None:
          # Load the region_index_to_country_index mapping
          json_response = requests.get(region_index_to_country_index_path)
          json_response.raise_for_status()
          
          print(f"Loaded {len(json.loads(json_response.text))} region to country index mappings.")
          
          # Load the file into a variable
          self.region_index_to_country_index = json.loads(json_response.text)
          # Convert the keys to integers
          self.region_index_to_country_index = {int(k): v for k, v in self.region_index_to_country_index.items()}
        
        self.test_loader = DataLoader(CustomImageDataset(images, coordinates, countries, regions, country_to_index=self.country_to_index, region_to_index=self.region_to_index), batch_size=batch_size, shuffle=False)
