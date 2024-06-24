import os
import sys
from torch.utils.data import DataLoader
from torchvision import transforms
sys.path.insert(0, '../')
from data_loader import load_image_files

sys.path.insert(0, '../5_evaluation')
from image_data_handler_inference import InferenceImageDataHandler

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

class DeployImageDataHandler(InferenceImageDataHandler):
    def __init__(self, country_to_index_path='./country_to_index.json', region_to_index_path='./region_to_index.json', region_index_to_middle_point_path='./region_index_to_middle_point.json', region_index_to_country_index_path='./region_index_to_country_index.json', base_transform=None, final_transform=None, max_batch_size=100):
        super().__init__(country_to_index_path, region_to_index_path, region_index_to_middle_point_path, region_index_to_country_index_path)
        self.base_transform = base_transform
        self.final_transform = final_transform
      
    def load_single_image(self, unscaled_image_path, base_transform, final_transform):
      images = load_image_files(os.path.join(CURRENT_DIR, unscaled_image_path))
      if self.base_transform is not None and self.final_transform is not None:
        transform=transforms.Compose([base_transform, final_transform])
        transformed_images = []
        for image in images:
            transformed_images.append(transform(image))
        return DataLoader(transformed_images, batch_size=1, shuffle=False)
      else:
        raise ValueError("base_transform and final_transform must be provided.")
