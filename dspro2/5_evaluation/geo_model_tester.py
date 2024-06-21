import torch
import sys
import requests
import io
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '../4_modeling')
from geo_model_trainer import GeoModelTrainer
from custom_haversine_loss import GeolocalizationLoss

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
        
        total_loss = 0.0
        total_metric = 0.0
        top1_correct = 0
        top3_correct = 0
        top5_correct = 0
        data_loader = self.test_dataloader
        middle_points = (torch.tensor(list(self.region_index_to_middle_point.values())).to(self.device) if self.region_index_to_middle_point is not None else torch.full((len(data_loader), 2), 0, dtype=torch.float64)) if self.use_regions else None
        
        with torch.no_grad():
            if self.use_coordinates:
              criterion = nn.MSELoss()
            elif self.use_regions:
              criterion = GeolocalizationLoss(temperature=75)
            else:
              criterion = nn.CrossEntropyLoss()
            
            for images, coordinates, country_indices, region_indices in data_loader:
                images = images.to(self.device)
                targets = coordinates.to(self.device) if self.use_coordinates else (country_indices.to(self.device) if self.use_regions else region_indices.to(self.device))
                outputs = self.model(images)
                probabilities = F.softmax(outputs, dim=1)

                loss = criterion(outputs, targets) if not self.use_regions else criterion(outputs, middle_points, coordinates)
                
                total_loss += loss.item() * images.size(0)

                if self.use_coordinates:
                    total_metric += self.mean_spherical_distance(outputs, targets).item() * images.size(0)
                if self.use_regions:
                    total_metric += self.mean_spherical_distance(middle_points[outputs.argmax(dim=1)], middle_points[targets]).item() * images.size(0)
                if not self.use_coordinates:
                    # Get the top 5 predictions for each image
                    _, predicted_top5 = probabilities.topk(5, 1, True, True)

                    # Calculate different accuracies
                    correct = predicted_top5.eq(targets.view(-1, 1).expand_as(predicted_top5))

                    top1_correct += correct[:, 0].sum().item()
                    top3_correct += correct[:, :3].sum().item()
                    top5_correct += correct[:, :5].sum().item()
                    if self.use_regions:
                        # Get the country for each region
                        target_countries = country_indices.to(self.device)
                        predicted_countries_top5 = torch.tensor([[self.region_index_to_country_index.get(region_index, -1) for region_index in top5] for top5 in predicted_top5]).to(self.device)
                        countries_correct = predicted_countries_top5.eq(target_countries.view(-1, 1).expand_as(predicted_countries_top5))
                        # Calculate different accuracies
                        top1_correct_country += countries_correct[:, 0].sum().item()
                        top3_correct_country += countries_correct[:, :3].sum().item()
                        top5_correct_country += countries_correct[:, :5].sum().item()

        avg_loss = total_loss / len(data_loader.dataset)
        
        if self.use_coordinates or self.use_regions:
            avg_metric = total_metric / len(data_loader.dataset)

        if not self.use_coordinates:
            top1_accuracy = top1_correct / len(data_loader.dataset)
            top3_accuracy = top3_correct / len(data_loader.dataset)
            top5_accuracy = top5_correct / len(data_loader.dataset)

        if self.use_coordinates:
            print(f"Test Loss: {avg_loss:.4f}, Test Distance: {avg_metric:.4f}")
        elif self.use_regions:
            top1_correct_country = top1_correct_country / len(data_loader.dataset)
            top3_correct_country = top3_correct_country / len(data_loader.dataset)
            top5_correct_country = top5_correct_country / len(data_loader.dataset)
            print(f"Test Loss: {avg_loss:.4f}, Test Distance: {avg_metric:.4f}, Test Top 1 Accuracy: {top1_accuracy:.4f}, Test Top 3 Accuracy: {top3_accuracy:.4f}, Test Top 5 Accuracy: {top5_accuracy:.4f}")
            print(f"Test Top 1 Accuracy (Country): {top1_correct_country:.4f}, Test Top 3 Accuracy (Country): {top3_correct_country:.4f}, Test Top 5 Accuracy (Country): {top5_correct_country:.4f}")
        else:
            print(f"Test Loss: {avg_loss:.4f}, Test Top 1 Accuracy: {top1_accuracy:.4f}, Test Top 3 Accuracy: {top3_accuracy:.4f}, Test Top 5 Accuracy: {top5_accuracy:.4f}")
