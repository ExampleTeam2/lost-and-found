import random
import os
import gc
import json
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, mobilenet_v2, efficientnet_b1, efficientnet_b3, efficientnet_b4, efficientnet_b7
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
from torchvision.models.efficientnet import EfficientNet_B1_Weights, EfficientNet_B3_Weights, EfficientNet_B4_Weights, EfficientNet_B7_Weights
import numpy as np

import wandb
import uuid
from shapely.geometry import Point


class GeoModelTrainer:
  def __init__(self, datasize, train_dataloader, val_dataloader, num_classes=2, predict_coordinates=False, country_to_index=None, regionHandler=None, test_data_path=None, predict_regions=True):
      self.num_classes = num_classes
      self.train_dataloader = train_dataloader
      self.val_dataloader = val_dataloader
      self.datasize = datasize
      self.use_coordinates = predict_coordinates
      self.use_regions = predict_regions
      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
      self.patience = 5
      self.model_type = None
      self.model = None
      self.country_to_index = country_to_index
      self.test_data_path = test_data_path
      self.regionHandler = regionHandler
      
  def set_seed(self, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
      
  def initialize_model(self, model_type):
      self.model_type = model_type
      # Initialize already with best available weights (currently alias for IMAGENET1K_V2)
      if self.model_type == 'resnet18':
          model = resnet18(weights=ResNet18_Weights.DEFAULT)
      elif self.model_type == 'resnet34':
          model = resnet34(weights=ResNet34_Weights.DEFAULT)
      elif self.model_type == 'resnet50':
          model = resnet50(weights=ResNet50_Weights.DEFAULT)
      elif self.model_type == 'resnet101':
          model = resnet101(weights=ResNet101_Weights.DEFAULT)
      elif self.model_type == 'resnet152':
          model = resnet152(weights=ResNet152_Weights.DEFAULT)
      elif self.model_type == 'mobilenet_v2':
          model = mobilenet_v2(weights='IMAGENET1K_V2')
      elif self.model_type == 'efficientnet_b1':
          model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
      elif self.model_type == 'efficientnet_b3':
          model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
      elif self.model_type == 'efficientnet_b4':
          model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
      elif self.model_type == 'efficientnet_b7':
          model = efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT)
      else:
          raise ValueError("Unsupported model type. Supported types are: resnet18, resnet34, resnet50, resnet101, resnet152.")

      if "resnet" in self.model_type:
          # Modify the final layer based on the number of classes
          model.fc = nn.Linear(model.fc.in_features, self.num_classes)
          # Initialize weights of the classifier
          nn.init.kaiming_normal_(model.fc.weight, mode='fan_out', nonlinearity='relu')
          nn.init.constant_(model.fc.bias, 0)
      elif "efficientnet" in self.model_type or self.model_type == "mobilenet_v2":
          if self.model_type == "mobilenet_v2":
              dropout = 0.2
              in_features = 1280
          elif "efficientnet" in self.model_type:
              dropout = 0.5
              if self.model_type == "efficientnet_b1":
                  in_features = 1280
              elif self.model_type == "efficientnet_b3":
                  in_features = 1536
              elif self.model_type == "efficientnet_b4":
                  in_features = 1792
              elif self.model_type == "efficientnet_b7":
                  in_features = 2560
          # Modify the final layer based on the number of classes
          model.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=False),
                nn.Linear(in_features=in_features, out_features=self.num_classes, bias=True)
          )
          # Initialize weights of the classifier
          nn.init.kaiming_normal_(model.classifier[1].weight, mode='fan_out', nonlinearity='relu')
          nn.init.constant_(model.classifier[1].bias, 0)
      return model
  
  def coordinates_to_cartesian(self, lon, lat, R=6371):
    # Convert degrees to radians
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)

    # Cartesian coordinates using numpy
    x = R * np.cos(lat_rad) * np.cos(lon_rad)
    y = R * np.cos(lat_rad) * np.sin(lon_rad)
    z = R * np.sin(lat_rad)
    return np.stack([x, y, z], axis=-1)  # Ensure the output is a numpy array with the correct shape

  def spherical_distance(self, cartesian1, cartesian2, R=6371.0):
      cartesian1 = cartesian1.to(cartesian2.device)
      dot_product = (cartesian1 * cartesian2).sum(dim=1)
      
      norms1 = cartesian1.norm(p=2, dim=1)
      norms2 = cartesian2.norm(p=2, dim=1)

      cos_theta = dot_product / (norms1 * norms2)
      cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
      
      theta = torch.acos(cos_theta)
      # curved distance -> "Bogenmass"
      distance = R * theta
      return distance
  
  def mean_spherical_distance(self, preds, targets):
      distances = self.spherical_distance(preds, targets)
      return distances.mean()

  def train(self):
      with wandb.init(reinit=True) as run:
          config = run.config
          self.set_seed(config.seed)
          
          # Set seeds, configure optimizers, losses, etc.
          best_val_metric = float('inf') if self.use_coordinates else 0
          patience_counter = 0

          # Rename run name and initialize parameters in model name
          model_name = f"model_{config.model_name}_lr_{config.learning_rate}_opt_{config.optimizer}_weightDecay_{config.weight_decay}_imgSize_{config.input_image_size}"
          run_name = model_name + f"_{uuid.uuid4()}"
          wandb.run.name = run_name

          # Initialize model, optimizer and criterion
          self.model = self.initialize_model(model_type=config.model_name).to(self.device)

          if self.use_coordinates:
            criterion = nn.MSELoss()
          elif self.use_regions:
            criterion = nn.MSELoss() # is not used
          else:
            criterion = nn.CrossEntropyLoss()

          if "resnet" in self.model_type:
              optimizer_grouped_parameters = [
                  {"params": [p for n, p in self.model.named_parameters() if not n.startswith('fc')], "lr": config.learning_rate * 0.1},
                  {"params": self.model.fc.parameters(), "lr": config.learning_rate}
              ]
          elif "efficientnet" in self.model_type or self.model_type == "mobilenet_v2":
              optimizer_grouped_parameters = [
                  {"params": [p for n, p in self.model.named_parameters() if not n.startswith('classifier')], "lr": config.learning_rate * 0.1},
                  {"params": self.model.classifier.parameters(), "lr": config.learning_rate}
                ]

          optimizer = optim.AdamW(optimizer_grouped_parameters, weight_decay=config.weight_decay)
          scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

          for epoch in range(config.epochs):
              if self.use_coordinates:
                  train_loss, train_metric = self.run_epoch(criterion, optimizer, is_train=True)
                  val_loss, val_metric = self.run_epoch(criterion, optimizer, is_train=False)
              else:
                  train_loss, train_top1_accuracy, train_top3_accuracy, train_top5_accuracy = self.run_epoch(criterion, optimizer, is_train=True)
                  val_loss, val_top1_accuracy, val_top3_accuracy, val_top5_accuracy = self.run_epoch(criterion, optimizer, is_train=False)

              if (self.use_coordinates and val_metric < best_val_metric) or (not self.use_coordinates and val_top1_accuracy > best_val_metric):
                  if self.use_coordinates:
                      best_val_metric = val_metric
                  else:
                      best_val_metric = val_top1_accuracy
                    
                  os.makedirs(f"models/datasize_{self.datasize}", exist_ok=True)
                  raw_model_path = f"best_model_checkpoint{model_name}_predict_coordinates_{self.use_coordinates}.pth"
                  model_path = f"models/datasize_{self.datasize}/{raw_model_path}"
                  torch.save(self.model.state_dict(), model_path)
                  patience_counter = 0
              else:
                  patience_counter += 1
                  if patience_counter >= self.patience:
                      print(f"Stopping early after {self.patience} epochs without improvement")
                      break
                      
              # Step the scheduler at the end of the epoch
              scheduler.step()

              # Log metrics to wandb
              if self.use_coordinates or self.use_regions:
                wandb.log({
                    "Train Loss": train_loss,
                    "Train Distance (km)": train_metric,
                    "Validation Loss": val_loss,
                    "Validation Distance (km)": val_metric
                })
              else:
                wandb.log({
                  "Train Loss": train_loss,
                  "Train Accuracy Top 1": train_top1_accuracy,
                  "Train Accuracy Top 3": train_top3_accuracy,
                  "Train Accuracy Top 5": train_top5_accuracy,
                  "Validation Loss": val_loss,
                  "Validation Accuracy Top 1": val_top1_accuracy,
                  "Validation Accuracy Top 3": val_top3_accuracy,
                  "Validation Accuracy Top 5": val_top5_accuracy
              })
                
          # Load and log the best model to wandb
          best_model = self.initialize_model(model_type=config.model_name).to(self.device)
          best_model.load_state_dict(torch.load(model_path))
          wandb_model_path = os.path.join(wandb.run.dir, raw_model_path)
          
          if self.country_to_index is not None:
              # copy to run directory
              wandb_country_to_index_file = os.path.join(wandb.run.dir, 'country_to_index.json')
              # write json file
              with open(wandb_country_to_index_file, 'w') as f:
                  json.dump(self.country_to_index, f)
              # save to wandb
              wandb.save(wandb_country_to_index_file)
              
          if self.test_data_path is not None:
            # Copy test data to run directory
            wandb_test_data_path = os.path.join(wandb.run.dir, 'test_data.pth')
            # write json file
            shutil.copy(self.test_data_path, wandb_test_data_path)
            wandb.save(wandb_test_data_path)
              
          torch.save(best_model.state_dict(), wandb_model_path)
          wandb.save(wandb_model_path)
          
          # Clean up
          del self.model
          del best_model

          gc.collect()
          torch.cuda.empty_cache()  

  def haversine_distance(self, coord1, coord2):
      """
      Calculate the Haversine distance between two points on the Earth specified in decimal degrees.
      """
      R = 6371.0  # Radius of the earth in kilometers

      print(f"coord1: {coord1}, coord2: {coord2}")

      # split the coordinates coord1 is a point and coord2 is a tensor of latitude and longitude
      lat1, lon1 = coord1.x, coord1.y
      lat2, lon2 = coord2[0], coord2[1]

      print(f"lat1: {lat1}, lon1: {lon1}, lat2: {lat2}, lon2: {lon2}")

      lat1, lon1, lat2, lon2 = map(lambda x: torch.tensor(x, dtype=torch.float64), [lat1, lon1, lat2, lon2])
      # Convert decimal degrees to radians 
      lat1, lon1, lat2, lon2 = map(torch.deg2rad, [lat1, lon1, lat2, lon2])

      # Haversine formula 
      dlat = lat2 - lat1 
      dlon = lon2 - lon1 
      a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
      c = 2 * torch.asin(torch.sqrt(a))
      distance = R * c  # Result in kilometers
      return distance
  
  #  haversine_smoothing loss function
  def haversine_smoothing_loss(self, outputs, targets, geocell_centroids, true_coords, tau=1.0):
      batch_size = outputs.size(0)
      num_classes = outputs.size(1)
      loss = 0.0
      for i in range(batch_size):
          for j in range(num_classes):
              geocell_centroid = geocell_centroids[j]
              true_geocell_centroid = geocell_centroids[targets[i].item()]

              d_true = self.haversine_distance(geocell_centroid, true_coords[i])
              d_pred = self.haversine_distance(true_geocell_centroid, true_coords[i])

              yn_i = torch.exp(-(d_true - d_pred) / tau)
              pn_i = outputs[i, j]

              loss += -torch.log(pn_i) * yn_i

      return loss

  def run_epoch(self, criterion, optimizer, is_train=True):
    if is_train:
        self.model.train()
    else:
        self.model.eval()

    total_loss = 0.0
    total_metric = 0.0
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    data_loader = self.train_dataloader if is_train else self.val_dataloader

    for images, coordinates, country_indices, region_indices in data_loader:
        with torch.set_grad_enabled(is_train):
            images = images.to(self.device)
            targets = coordinates.to(self.device) if self.use_coordinates else (country_indices.to(self.device) if self.use_regions else region_indices.to(self.device))
            optimizer.zero_grad()
            outputs = self.model(images)
            probabilities = F.softmax(outputs, dim=1)
            loss = criterion(outputs, targets) if not self.use_regions else self.haversine_smoothing_loss(outputs, targets, self.regionHandler.region_middle_points, coordinates)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            if self.use_coordinates:
                total_metric += self.mean_spherical_distance(outputs, targets).item() * images.size(0)
            elif self.use_regions:
                total_metric += loss.item() * images.size(0)
            else:
                # Get the top 5 predictions for each image
                _, predicted_top5 = probabilities.topk(5, 1, True, True)

                # Calculate different accuracies
                correct = predicted_top5.eq(targets.view(-1, 1).expand_as(predicted_top5))

                top1_correct += correct[:, 0].sum().item()
                top3_correct += correct[:, :3].sum().item()
                top5_correct += correct[:, :5].sum().item()

    avg_loss = total_loss / len(data_loader.dataset)

    if self.use_coordinates or self.use_regions:
        avg_metric = total_metric / len(data_loader.dataset)
        return avg_loss, avg_metric
    else:
        top1_accuracy = top1_correct / len(data_loader.dataset)
        top3_accuracy = top3_correct / len(data_loader.dataset)
        top5_accuracy = top5_correct / len(data_loader.dataset)
        return avg_loss, top1_accuracy, top3_accuracy, top5_accuracy
