import torch
import sys
import requests
import io

sys.path.insert(0, "../4_modeling")
from geo_model_harness import GeoModelHarness


class GeoModelInference(GeoModelHarness):
    def __init__(self, num_classes=3, predict_coordinates=False, country_to_index=None, region_to_index=None, region_index_to_middle_point=None, region_index_to_country_index=None, predict_regions=False):
        super().__init__(num_classes=num_classes, predict_coordinates=predict_coordinates, country_to_index=country_to_index, region_to_index=region_to_index, region_index_to_middle_point=region_index_to_middle_point, region_index_to_country_index=region_index_to_country_index, predict_regions=predict_regions)

    def prepare(self, model_type, model_path):
        # Load the model
        self.initialize_model(model_type=model_type)
        model_response = requests.get(model_path)
        model_response.raise_for_status()
        pretrained_weights = io.BytesIO(model_response.content)
        self.model.load_state_dict(torch.load(pretrained_weights, map_location=self.device))

        self.model.eval()
