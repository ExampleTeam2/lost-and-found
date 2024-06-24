import sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '../5_evaluation')
from image_data_handler_inference import InferenceImageDataHandler

class DeployImageDataHandler(InferenceImageDataHandler):
    def __init__(self, country_to_index_path='./country_to_index.json', region_to_index_path='./region_to_index.json', region_index_to_middle_point_path='./region_index_to_middle_point.json', region_index_to_country_index_path='./region_index_to_country_index.json', max_batch_size=100):
        super().__init__(country_to_index_path, region_to_index_path, region_index_to_middle_point_path, region_index_to_country_index_path)
      
    def load_single_image(self, unscaled_image_path):
      images = [torch.tensor(0)]
      # TODO: Implement
      return DataLoader(images, batch_size=1, shuffle=False)
