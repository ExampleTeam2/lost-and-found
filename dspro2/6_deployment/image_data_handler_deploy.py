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
    def __init__(self, country_to_index_path='./country_to_index.json', region_to_index_path='./region_to_index.json', region_index_to_middle_point_path='./region_index_to_middle_point.json', region_index_to_country_index_path='./region_index_to_country_index.json', base_transform=None, max_batch_size=100):
        super().__init__(country_to_index_path, region_to_index_path, region_index_to_middle_point_path, region_index_to_country_index_path)
        self.base_transform = base_transform
        self.max_batch_size = max_batch_size
        
    def load_images(self, unscaled_image_paths):
      paths = [os.path.join(CURRENT_DIR, path) for path in unscaled_image_paths]
      images = load_image_files(paths)
      # if aspect ratio is not 16:9, crop the image
      for i, image in enumerate(images):
        if image.size[0] / image.size[1] != 16 / 9:
          print("Cropping image to 16:9 aspect ratio.")
          images[i] = image.crop((0, 0, image.size[0], int(image.size[0] * 9 / 16)))
      if self.base_transform is not None:
        transform=self.base_transform
        transformed_images = []
        for image in images:
            transformed_images.append(transform(image))
        return DataLoader(transformed_images, batch_size=min(len(unscaled_image_paths), self.max_batch_size), shuffle=False)
      else:
        raise ValueError("base_transform must be provided.")
      
    def load_single_image(self, unscaled_image_path):
      return self.load_images([unscaled_image_path])
