import torch
import json
import os
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, images, coordinates, countries, datasize, country_to_index=None, replace_country_index=False, hash=None):
        self.images = images
        self.coordinates = coordinates
        self.countries = countries
        self.country_index_path = f"models/datasize_{datasize}_country_to_index.json" if not hash else f"models/datasize_{datasize}_{hash}_country_to_index.json"

        if replace_country_index or country_to_index is None:
            # Create a new country_to_index mapping
            unique_countries = sorted(set(countries))
            self.country_to_index = {country: idx for idx, country in enumerate(unique_countries)}
            with open(self.country_index_path, 'w') as f:
                json.dump(self.country_to_index, f)
        else:
            # Use the provided country_to_index mapping
            self.country_to_index = country_to_index

        # Debugging: print the mapping and check for missing countries
        missing_countries = set(self.countries) - set(self.country_to_index.keys())
        if missing_countries:
            print(f"Warning: The following countries are missing in the mapping: {missing_countries}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        country = self.countries[idx]
        if country not in self.country_to_index:
            raise ValueError(f"Country '{country}' at index {idx} is not in the country_to_index mapping.")
        country_index = self.country_to_index[country]
        coordinates = torch.tensor(self.coordinates[idx], dtype=torch.float32)

        return image, coordinates, country_index
