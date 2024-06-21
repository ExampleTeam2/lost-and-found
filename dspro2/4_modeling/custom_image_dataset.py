import torch
import json
import os
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, images, coordinates, countries, regions, country_to_index, region_to_index):
        self.images = images
        self.coordinates = coordinates
        self.countries = countries
        self.regions = regions
        self.country_to_index = country_to_index
        self.region_to_index = region_to_index

        # Debugging: print the mapping and check for missing countries
        missing_countries = set(self.countries) - set(self.country_to_index.keys())
        if missing_countries:
            print(f"Warning: The following countries are missing in the mapping: {missing_countries}")


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        country = self.countries[idx]
        region = self.regions[idx]
        if country not in self.country_to_index:
            raise ValueError(f"Country '{country}' at index {idx} is not in the country_to_index mapping.")
        country_index = self.country_to_index[country] if self.country_to_index is not None else 0
        region_index = self.region_to_index[region[0]] if self.region_to_index is not None else 0
        coordinates = torch.tensor(self.coordinates[idx], dtype=torch.float32)

        return image, coordinates, country_index, region_index
