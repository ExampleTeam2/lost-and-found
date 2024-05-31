import sys
import random
import json
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from custom_image_name_dataset import CustomImageNameDataset
from custom_image_dataset import CustomImageDataset

sys.path.insert(0, '../')
from data_loader import split_json_and_image_files, load_json_files, load_image_files, potentially_get_cached_file_path, get_cached_file_path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TestImageDataHandler:
  def __init__(self, test_path='./test_data.pth', country_to_index_path='./country_to_index.json', batch_size=100):
    print(f"Loading test data from {os.path.basename(test_path)}")
    test_data = torch.load(test_path)
    print("Test data loaded.")
    images, countries, coordinates = test_data['test_images'], test_data['test_countries'], test_data['test_coordinates']
        
    with open(country_to_index_path, 'r') as f:
      self.country_to_index = json.load(f)
    
    self.test_loader = DataLoader(CustomImageDataset(images, coordinates, countries, country_to_index=self.country_to_index), batch_size=batch_size, shuffle=False)

class ImageDataHandler:
    def __init__(self, list_files, base_transform, augmented_transform, final_transform, preprocessing_config={}, batch_size=100, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, cache=True, cache_zip_load_callback=None, cache_pth_save_callback=None, save_test_data=True, random_seed=42):
        assert train_ratio + val_ratio + test_ratio - 1 <= 0.001, "Ratios should sum to 1"
          
        self.batch_size = batch_size
        self.random_seed = random_seed
        
        json_paths, image_paths = split_json_and_image_files(list_files)
        
        combined_names = list(zip(image_paths, json_paths))
        total_count = len(combined_names)
        train_end = int(train_ratio * total_count)
        val_end = train_end + int(val_ratio * total_count)
        train_images, train_jsons = list(zip(*combined_names[:train_end]))
        val_images, val_jsons = list(zip(*combined_names[train_end:val_end]))
        test_images, test_jsons = list(zip(*combined_names[val_end:]))
      
        train_file_name_dataset = CustomImageNameDataset(train_images, train_jsons)
        val_file_name_dataset = CustomImageNameDataset(val_images, val_jsons)
        test_file_name_dataset = CustomImageNameDataset(test_images, test_jsons)
        
        self.train_images = []
        self.val_images = []
        self.test_images = []
        self.train_countries = []
        self.val_countries = []
        self.test_countries = []
        self.train_coordinates = []
        self.val_coordinates = []
        self.test_coordinates = []
        
        cached_data = potentially_get_cached_file_path(list_files, preprocessing_config) if cache else None
        if cache and cached_data is None and cache_zip_load_callback is not None:
          cache_zip_load_callback()
        if cached_data is not None:
          
          print(f"Using cached data from: {os.path.basename(cached_data)}")
          data = torch.load(cached_data)
          print("Data loaded.")
          self.train_images = data['train_images']
          self.val_images = data['val_images']
          self.test_images = data['test_images']
          self.train_countries = data['train_countries']
          self.val_countries = data['val_countries']
          self.test_countries = data['test_countries']
          self.train_coordinates = data['train_coordinates']
          self.val_coordinates = data['val_coordinates']
          self.test_coordinates = data['test_coordinates']
          del data
          
        else:
          
          train_file_name_loader = DataLoader(train_file_name_dataset, batch_size=batch_size, shuffle=False)
          val_file_name_loader = DataLoader(val_file_name_dataset, batch_size=batch_size, shuffle=False)
          test_file_name_loader = DataLoader(test_file_name_dataset, batch_size=batch_size, shuffle=False)
          
          train_transform=transforms.Compose([base_transform, augmented_transform, final_transform] if augmented_transform is not None else [base_transform, final_transform])
          val_transform=transforms.Compose([base_transform, final_transform])
          test_transform=transforms.Compose([base_transform, final_transform])
          
          random.seed(random_seed)
          torch.manual_seed(random_seed)
          for batch_image_files, batch_label_files in tqdm(train_file_name_loader, desc="Loading train images and labels"):
              images = load_image_files(batch_image_files)
              labels = load_json_files(batch_label_files)
              self.train_countries.extend([item['country_name'] for item in labels])
              self.train_coordinates.extend([item['coordinates'] for item in labels])
              for image in images:
                  self.train_images.append(train_transform(image))
              
          random.seed(random_seed)
          torch.manual_seed(random_seed)    
          for batch_image_files, batch_label_files in tqdm(val_file_name_loader, desc="Loading val images and labels"):
              images = load_image_files(batch_image_files)
              labels = load_json_files(batch_label_files)
              self.val_countries.extend([item['country_name'] for item in labels])
              self.val_coordinates.extend([item['coordinates'] for item in labels])
              for image in images:
                  self.val_images.append(val_transform(image))
                  
          random.seed(random_seed)
          torch.manual_seed(random_seed)
          for batch_image_files, batch_label_files in tqdm(test_file_name_loader, desc="Loading test images and labels"):
              images = load_image_files(batch_image_files)
              labels = load_json_files(batch_label_files)
              self.test_countries.extend([item['country_name'] for item in labels])
              self.test_coordinates.extend([item['coordinates'] for item in labels])
              for image in images:
                  self.test_images.append(test_transform(image))
                  
          if cache:
            data = {
                'train_images': self.train_images,
                'val_images': self.val_images,
                'test_images': self.test_images,
                'train_countries': self.train_countries,
                'val_countries': self.val_countries,
                'test_countries': self.test_countries,
                'train_coordinates': self.train_coordinates,
                'val_coordinates': self.val_coordinates,
                'test_coordinates': self.test_coordinates
            }
            print("Caching data...")
            torch.save(data, get_cached_file_path(list_files, preprocessing_config))
            print("Data cached.")
            del data
            if cache_pth_save_callback is not None:
              cache_pth_save_callback()
              
        self.test_data_path = None
        
        if save_test_data:
          test_data = {
            'test_images': self.test_images,
            'test_countries': self.test_countries,
            'test_coordinates': self.test_coordinates
          }
          self.test_data_path = './test_data.pth'
          print(f"Saving test data to {os.path.basename(self.test_data_path)}")
          torch.save(test_data, self.test_data_path)
          print("Test data saved.")
          del test_data
                
        self.countries = [*self.train_countries, *self.val_countries, *self.test_countries]
        
        # Create a global country_to_index mapping
        self.country_to_index = self._get_country_to_index()
        
        # Initialize datasets and loaders
        self.train_loader, self.val_loader, self.test_loader = self._create_loaders()
        
    def _get_country_to_index(self):
        # Gather all unique countries and create a global country_to_index mapping
        all_countries = set(self.countries)
        country_to_index = {country: idx for idx, country in enumerate(sorted(all_countries))}
        return country_to_index

    def _create_loaders(self):
        # Create train, val, and test datasets with the same mapping
        train_dataset = CustomImageDataset(self.train_images, self.train_coordinates, self.train_countries, country_to_index=self.country_to_index)
        val_dataset = CustomImageDataset(self.val_images, self.val_coordinates, self.val_countries, country_to_index=self.country_to_index)
        test_dataset = CustomImageDataset(self.test_images, self.test_coordinates, self.test_countries, country_to_index=self.country_to_index)

        # Create train, val, and test dataloaders
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
      