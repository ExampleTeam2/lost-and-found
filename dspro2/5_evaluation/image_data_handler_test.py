import sys
import json
import os
import torch
from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sys.path.insert(0, '../4_modeling')
from custom_image_dataset import CustomImageDataset
from region_handler import RegionHandler

class TestImageDataHandler:
  def __init__(self, test_path='./test_data.pth', country_to_index_path='./country_to_index.json', batch_size=100):
    print(f"Loading test data from {os.path.basename(test_path)}")
    test_data = torch.load(test_path)
    print("Test data loaded.")
    images, countries, coordinates, regions = test_data['test_images'], test_data['test_countries'], test_data['test_coordinates'], test_data['test_regions']
        
    with open(country_to_index_path, 'r') as f:
      self.country_to_index = json.load(f)
    
    self.region_handler = RegionHandler()
    
    self.test_loader = DataLoader(CustomImageDataset(images, coordinates, countries, regions, country_to_index=self.country_to_index, region_to_index=self.region_handler), batch_size=batch_size, shuffle=False)
