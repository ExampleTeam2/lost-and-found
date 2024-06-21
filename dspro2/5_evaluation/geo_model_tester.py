import torch
import sys
import requests
import io
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '../4_modeling')
from geo_model_trainer import GeoModelTrainer

class GeoModelTester(GeoModelTrainer):
  def __init__(self, datasize, train_dataloader, val_dataloader, test_dataloader, num_classes=2, predict_coordinates=False, country_to_index=None, region_to_index=None, region_index_to_middle_point=None, region_index_to_country_index=None, test_data_path=None, predict_regions=False):
      super().__init__(datasize, train_dataloader, val_dataloader, num_classes, predict_coordinates, country_to_index, region_to_index, region_index_to_middle_point, region_index_to_country_index, test_data_path, predict_regions)
      self.test_dataloader = test_dataloader

  def test(self, model_type, model_path):
        # Load the model
        self.model = self.initialize_model(model_type=model_type).to(self.device)
        model_response = requests.get(model_path)
        model_response.raise_for_status()
        pretrained_weights = io.BytesIO(model_response.content)
        self.model.load_state_dict(torch.load(pretrained_weights, map_location=self.device))

        self.model.eval()
        
        with torch.no_grad():
        
          if self.use_coordinates:
              test_loss, test_metric = self.run_epoch(optimizer=None, is_train=False, is_test=True)
          elif self.use_regions:
              test_loss, test_metric, test_top1_accuracy, test_top3_accuracy, test_top5_accuracy, test_top1_correct_country, test_top3_correct_country, test_top5_correct_country = self.run_epoch(optimizer=None, is_train=False, is_test=True)
          else:
              test_loss, test_top1_accuracy, test_top3_accuracy, test_top5_accuracy = self.run_epoch(optimizer=None, is_train=False, is_test=True)

        if self.use_coordinates:
            print(f"Test Loss: {test_loss:.4f}, Test Distance: {test_metric:.4f}")
        elif self.use_regions:
            print(f"Test Loss: {test_loss:.4f}, Test Distance: {test_metric:.4f}, Test Top 1 Accuracy: {test_top1_accuracy:.4f}, Test Top 3 Accuracy: {test_top3_accuracy:.4f}, Test Top 5 Accuracy: {test_top5_accuracy:.4f}")
            print(f"Test Top 1 Accuracy (Country): {test_top1_correct_country:.4f}, Test Top 3 Accuracy (Country): {test_top3_correct_country:.4f}, Test Top 5 Accuracy (Country): {test_top5_correct_country:.4f}")
        else:
            print(f"Test Loss: {test_loss:.4f}, Test Top 1 Accuracy: {test_top1_accuracy:.4f}, Test Top 3 Accuracy: {test_top3_accuracy:.4f}, Test Top 5 Accuracy: {test_top5_accuracy:.4f}")
