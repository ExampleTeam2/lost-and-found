import torch
import json
import os
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, images, coordinates, countries, datasize, replace_country_index):
        self.images = images
        self.coordinates = coordinates
        self.countries = countries
        self.country_index_path= f"models/datasize_{datasize}_country_to_index.json"
        absolute_path = os.path.abspath(self.country_index_path)

        print(f"Country index path (relative): {self.country_index_path}")
        print(f"Country index path (absolute): {absolute_path}")

        if replace_country_index:
          # Try to load existing country to index mapping from a JSON file
          with open(self.country_index_path, 'r') as f:
              self.country_to_index = json.load(f)
          # Ensure keys read from JSON (which will be strings) are correctly handled if numeric
          if all(isinstance(key, str) and key.isdigit() for key in self.country_to_index.keys()):
              # Convert keys from strings to integers if necessary
              self.country_to_index = {int(k): v for k, v in self.country_to_index.items()}
        else:
            unique_countries = sorted(set(countries))
            self.country_to_index = {country: idx for idx, country in enumerate(unique_countries)}
            with open(self.country_index_path, 'w') as f:
                  json.dump(self.country_to_index, f)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        country_index = self.country_to_index[self.countries[idx]]
        coordinates = torch.tensor(self.coordinates[idx], dtype=torch.float32)

        return image, coordinates, country_index
