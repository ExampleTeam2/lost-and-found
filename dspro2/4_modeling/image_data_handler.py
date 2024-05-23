import sys
import random
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from custom_image_name_dataset import CustomImageNameDataset
from custom_image_dataset import CustomImageDataset

sys.path.insert(0, '../')
from data_loader import load_json_files, load_image_files
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ImageDataHandler:
    def __init__(self, image_paths, json_paths, transform, datasize, batch_size=100, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        self.batch_size = batch_size
        self.datasize = datasize
      
        file_name_dataset = CustomImageNameDataset(image_paths, json_paths, transform=transform)
        file_name_loader = DataLoader(file_name_dataset, batch_size=batch_size, shuffle=False)
        
        self.images = []
        self.countries = []
        self.coordinates = []

        for batch_image_files, batch_label_files in tqdm(file_name_loader, desc="Loading images and labels"):
            images = load_image_files(batch_image_files)
            labels = load_json_files(batch_label_files)
            self.countries.extend([item['country_name'] for item in labels])
            #self.countries.extend([item.get('country_name', 'Unknown') for item in labels])
            self.coordinates.extend([item['coordinates'] for item in labels])
            #self.coordinates.extend([item.get('coordinates', 'Unknown') for item in labels])
            for image in images:
                self.images.append(transform(image))
                #self.images.append(transform(image).to(device))
        
        # Initialize datasets and loaders
        self.train_loader, self.val_loader, self.test_loader = self.create_loaders(train_ratio, val_ratio, test_ratio)

    def create_loaders(self, train_ratio, val_ratio, test_ratio):
        assert train_ratio + val_ratio + test_ratio - 1 <= 0.001, "Ratios should sum to 1"
        
        combined = list(zip(self.images, self.coordinates, self.countries))
        random.shuffle(combined)
        total_count = len(combined)
        train_end = int(train_ratio * total_count)
        val_end = train_end + int(val_ratio * total_count)

        train_data = combined[:train_end]
        val_data = combined[train_end:val_end]
        test_data = combined[val_end:]
        
        # Create train, val- and test datasets
        train_dataset = CustomImageDataset(*zip(*train_data), self.datasize, replace_country_index=True)
        val_dataset = CustomImageDataset(*zip(*val_data), self.datasize, replace_country_index=False)
        test_dataset = CustomImageDataset(*zip(*test_data), self.datasize, replace_country_index=False)

        # Create train, val- and test dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        return train_loader, val_loader, test_loader
