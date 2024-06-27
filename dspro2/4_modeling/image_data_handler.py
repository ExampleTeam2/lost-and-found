import sys
import random
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from custom_image_name_dataset import CustomImageNameDataset
from custom_image_dataset import CustomImageDataset
from region_handler import RegionHandler

sys.path.insert(0, "../")
from data_loader import split_json_and_image_files, load_json_files, load_image_files, potentially_get_cached_file_path, get_cached_file_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Helper function to inspect transformed images, back to PIL format and display
def inspect_transformed_images(transformed_images, num_images=5):
    for i in range(num_images):
        img = transformed_images[i]
        img = transforms.ToPILImage()(img)
        img.show()


class ImageDataHandler:
    def __init__(self, list_files, augmented_transform, base_transform, preprocessing_config={}, prediction_type=None, batch_size=100, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, cache=True, cache_zip_load_callback=None, cache_additional_save_callback=None, save_test_data=True, random_seed=42, inspect_transformed=False, move_files=False):
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
        self.train_regions = []
        self.val_regions = []
        self.test_regions = []

        cached_data = potentially_get_cached_file_path(list_files, preprocessing_config) if cache else None
        if cache and cached_data is None and cache_zip_load_callback is not None:
            cache_zip_load_callback()
        if cached_data is not None:

            print(f"Using cached data from: {os.path.basename(cached_data)}")
            data = torch.load(cached_data)
            print("Data loaded.")
            self.train_images = data["train_images"]
            self.val_images = data["val_images"]
            self.test_images = data["test_images"]
            self.train_countries = data["train_countries"]
            self.val_countries = data["val_countries"]
            self.test_countries = data["test_countries"]
            self.train_coordinates = data["train_coordinates"]
            self.val_coordinates = data["val_coordinates"]
            self.test_coordinates = data["test_coordinates"]
            self.train_regions = data["train_regions"]
            self.val_regions = data["val_regions"]
            self.test_regions = data["test_regions"]
            del data

        else:

            train_file_name_loader = DataLoader(train_file_name_dataset, batch_size=batch_size, shuffle=False)
            val_file_name_loader = DataLoader(val_file_name_dataset, batch_size=batch_size, shuffle=False)
            test_file_name_loader = DataLoader(test_file_name_dataset, batch_size=batch_size, shuffle=False)

            train_transform = transforms.Compose([augmented_transform, base_transform]) if augmented_transform is not None else base_transform
            val_transform = base_transform
            test_transform = base_transform

            random.seed(random_seed)
            torch.manual_seed(random_seed)
            for batch_image_files, batch_label_files in tqdm(train_file_name_loader, desc="Loading train images and labels"):
                images = load_image_files(batch_image_files)
                labels = load_json_files(batch_label_files)
                self.train_countries.extend([item["country_name"] for item in labels])
                self.train_coordinates.extend([item["coordinates"] for item in labels])
                self.train_regions.extend([item["regions"][0] for item in labels])
                for image in images:
                    self.train_images.append(train_transform(image))

            if inspect_transformed:
                inspect_transformed_images(self.train_images, num_images=5)

            random.seed(random_seed)
            torch.manual_seed(random_seed)
            for batch_image_files, batch_label_files in tqdm(val_file_name_loader, desc="Loading val images and labels"):
                images = load_image_files(batch_image_files)
                labels = load_json_files(batch_label_files)
                self.val_countries.extend([item["country_name"] for item in labels])
                self.val_coordinates.extend([item["coordinates"] for item in labels])
                self.val_regions.extend([item["regions"][0] for item in labels])
                for image in images:
                    self.val_images.append(val_transform(image))

            random.seed(random_seed)
            torch.manual_seed(random_seed)
            for batch_image_files, batch_label_files in tqdm(test_file_name_loader, desc="Loading test images and labels"):
                images = load_image_files(batch_image_files)
                labels = load_json_files(batch_label_files)
                self.test_countries.extend([item["country_name"] for item in labels])
                self.test_coordinates.extend([item["coordinates"] for item in labels])
                self.test_regions.extend([item["regions"][0] for item in labels])
                for image in images:
                    self.test_images.append(test_transform(image))

            if cache:
                data = {"train_images": self.train_images, "val_images": self.val_images, "test_images": self.test_images, "train_countries": self.train_countries, "val_countries": self.val_countries, "test_countries": self.test_countries, "train_coordinates": self.train_coordinates, "val_coordinates": self.val_coordinates, "test_coordinates": self.test_coordinates, "train_regions": self.train_regions, "val_regions": self.val_regions, "test_regions": self.test_regions}
                print("Caching data...")
                torch.save(data, get_cached_file_path(list_files, preprocessing_config))
                print("Data cached.")
                del data
                if cache_additional_save_callback is not None:
                    cache_additional_save_callback(move=move_files)

        self.test_data_path = None
        self.run_link = None
        self.run_link_path = None

        if save_test_data:
            # If previous run already saved test data, use that run link
            cached_run_link = potentially_get_cached_file_path(list_files, {"prediction_type": prediction_type, **preprocessing_config} if prediction_type is not None else preprocessing_config, "run", ".wandb")
            if cached_run_link is not None:
                print(f"Using run link from: {os.path.basename(cached_run_link)}")
                # Read first line as text, strip whitespace, and split by newline
                with open(cached_run_link, "r") as f:
                    self.run_link = f.readline().strip()
                print("Skipping test data saving.")

            else:
                self.run_link_path = get_cached_file_path(list_files, {"prediction_type": prediction_type, **preprocessing_config} if prediction_type is not None else preprocessing_config, "run", ".wandb")
                print(f"Creating new run link at {os.path.basename(self.run_link_path)}")

                test_data = {"test_images": self.test_images, "test_countries": self.test_countries, "test_coordinates": self.test_coordinates, "test_regions": self.test_regions}
                self.test_data_path = "./test_data.pth"
                print(f"Saving test data to {os.path.basename(self.test_data_path)}")
                torch.save(test_data, self.test_data_path)
                print("Test data saved.")
                del test_data

        self.countries = [*self.train_countries, *self.val_countries, *self.test_countries]

        # Create a global country_to_index mapping
        self.country_to_index = self._get_country_to_index()

        self.region_handler = RegionHandler(country_to_index=self.country_to_index)

        # Get the global region_to_index mapping
        self.region_to_index = self.region_handler.region_to_index
        # Get the global region_index_to_middle_point mapping
        self.region_index_to_middle_point = self.region_handler.region_index_to_middle_point
        # Get the global region_index_to_country_index mapping
        self.region_index_to_country_index = self.region_handler.region_index_to_country_index

        # Initialize datasets and loaders
        self.train_loader, self.val_loader, self.test_loader = self._create_loaders()

    def _get_country_to_index(self):
        # Gather all unique countries and create a global country_to_index mapping
        all_countries = set(self.countries)
        country_to_index = {country: idx for idx, country in enumerate(sorted(all_countries))}
        return country_to_index

    def _create_loaders(self):
        # Create train, val, and test datasets with the same mapping
        train_dataset = CustomImageDataset(self.train_images, self.train_coordinates, self.train_countries, self.train_regions, country_to_index=self.country_to_index, region_to_index=self.region_to_index)
        val_dataset = CustomImageDataset(self.val_images, self.val_coordinates, self.val_countries, self.val_regions, country_to_index=self.country_to_index, region_to_index=self.region_to_index)
        test_dataset = CustomImageDataset(self.test_images, self.test_coordinates, self.test_countries, self.test_regions, country_to_index=self.country_to_index, region_to_index=self.region_to_index)

        # Create train, val, and test dataloaders
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
