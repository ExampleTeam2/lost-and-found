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
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, efficientnet_b1, efficientnet_b3, efficientnet_b4, efficientnet_b7
from torchvision.models import mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
from torchvision.models.efficientnet import EfficientNet_B1_Weights, EfficientNet_B3_Weights, EfficientNet_B4_Weights, EfficientNet_B7_Weights
import numpy as np
import wandb
import uuid
from shapely.geometry import Point
from sklearn.metrics import balanced_accuracy_score

from custom_haversine_loss import GeolocalizationLoss


class GeoModelTrainer:
  def __init__(self, datasize, train_dataloader, val_dataloader, num_classes=2, predict_coordinates=False, country_to_index=None, region_to_index=None, region_index_to_middle_point=None, region_index_to_country_index=None, test_data_path=None, predict_regions=False, run_start_callback=None):
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
      self.index_to_country = {v: k for k, v in country_to_index.items()} if country_to_index is not None else None
      self.region_to_index = region_to_index
      self.index_to_region = {v: k for k, v in region_to_index.items()} if region_to_index is not None else None
      self.region_index_to_middle_point = region_index_to_middle_point
      self.region_index_to_country_index = region_index_to_country_index
      self.test_data_path = test_data_path
      self.run_start_callback = run_start_callback
      
      # For training
      
      self.region_middle_points = (torch.tensor(list(self.region_index_to_middle_point.values())).to(self.device) if self.region_index_to_middle_point is not None else torch.tensor([], dtype=torch.float64)) if self.use_regions else None
      
      if self.use_coordinates:
        self.criterion = nn.MSELoss()
      elif self.use_regions:
        self.criterion = GeolocalizationLoss(temperature=75)
      else:
        self.criterion = nn.CrossEntropyLoss()
      
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
      elif self.model_type == 'mobilenet_v3_small':
          model = mobilenet_v3_small(weights='IMAGENET1K_V1')
      elif self.model_type == 'mobilenet_v3_large':
          model = mobilenet_v3_large(weights='IMAGENET1K_V1')
      elif self.model_type == 'efficientnet_b1':
          model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
      elif self.model_type == 'efficientnet_b3':
          model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
      elif self.model_type == 'efficientnet_b4':
          model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
      elif self.model_type == 'efficientnet_b7':
          model = efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT)
      else:
          raise ValueError("Unsupported model type. Supported types are: resnet18, resnet34, resnet50, resnet101, resnet152, mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large, efficientnet_b1, efficientnet_b3, efficientnet_b4, efficientnet_b7")

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
      elif "mobilenet_v3" in self.model_type:
          if self.model_type == "mobilenet_v3_small":
              in_features = 1024
          else:
              in_features = 1280
          # Modify the final layer based on the number of classes
          model.classifier[3] = torch.nn.Linear(in_features=in_features, out_features=self.num_classes, bias=True)
          # Initialize weights of the classifier
          nn.init.kaiming_normal_(model.classifier[3].weight, mode='fan_out', nonlinearity='relu')
          nn.init.constant_(model.classifier[3].bias, 0)
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
          
          if self.run_start_callback is not None:
              self.run_start_callback(config, run)
          
          self.set_seed(config.seed)
          
          # Set seeds, configure optimizers, losses, etc.
          best_val_metric = float('inf') if self.use_coordinates or self.use_regions else 0
          patience_counter = 0

          # Rename run name and initialize parameters in model name
          model_name = f"model_{config.model_name}_lr_{config.learning_rate}_opt_{config.optimizer}_weightDecay_{config.weight_decay}_imgSize_{config.input_image_size}"
          run_name = model_name + f"_{uuid.uuid4()}"
          wandb.run.name = run_name

          # Initialize model, optimizer and criterion
          self.model = self.initialize_model(model_type=config.model_name).to(self.device)

          if "resnet" in self.model_type:
              optimizer_grouped_parameters = [
                  {"params": [p for n, p in self.model.named_parameters() if not n.startswith('fc')], "lr": config.learning_rate * 0.1},
                  {"params": self.model.fc.parameters(), "lr": config.learning_rate}
              ]
          elif "efficientnet" in self.model_type or "mobilenet" in self.model_type:
              optimizer_grouped_parameters = [
                  {"params": [p for n, p in self.model.named_parameters() if not n.startswith('classifier')], "lr": config.learning_rate * 0.1},
                  {"params": self.model.classifier.parameters(), "lr": config.learning_rate}
                ]

          optimizer = optim.AdamW(optimizer_grouped_parameters, weight_decay=config.weight_decay)
          scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
          
          for epoch in range(config.epochs):
              if self.use_coordinates:
                  train_loss, train_metric = self.run_epoch(optimizer, is_train=True)
                  val_loss, val_metric = self.run_epoch(optimizer, is_train=False)
              elif self.use_regions:
                  train_loss, train_metric, train_top1_accuracy, train_top3_accuracy, train_top5_accuracy, train_top1_correct_country, train_top3_correct_country, train_top5_correct_country = self.run_epoch(optimizer, is_train=True)
                  val_loss, val_metric, val_top1_accuracy, val_top3_accuracy, val_top5_accuracy, val_top1_correct_country, val_top3_correct_country, val_top5_correct_country = self.run_epoch(optimizer, is_train=False)
              else:
                  train_loss, train_top1_accuracy, train_top3_accuracy, train_top5_accuracy = self.run_epoch(optimizer, is_train=True)
                  val_loss, val_top1_accuracy, val_top3_accuracy, val_top5_accuracy = self.run_epoch(optimizer, is_train=False)

              # Even for predicting regions, always use the best model based on validation distance
              if ((self.use_coordinates or self.use_regions) and val_metric < best_val_metric) or ((not (self.use_coordinates or self.use_regions)) and val_top1_accuracy > best_val_metric):
                  if self.use_coordinates or self.use_regions:
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
              if self.use_coordinates:
                wandb.log({
                    "Train Loss": train_loss,
                    "Train Distance (km)": train_metric,
                    "Validation Loss": val_loss,
                    "Validation Distance (km)": val_metric
                })
              elif self.use_regions:
                wandb.log({
                    "Train Loss": train_loss,
                    "Train Distance (km)": train_metric,
                    "Train Accuracy Top 1": train_top1_accuracy,
                    "Train Accuracy Top 3": train_top3_accuracy,
                    "Train Accuracy Top 5": train_top5_accuracy,
                    "Train Accuracy Top 1 Country": train_top1_correct_country,
                    "Train Accuracy Top 3 Country": train_top3_correct_country,
                    "Train Accuracy Top 5 Country": train_top5_correct_country,
                    "Validation Loss": val_loss,
                    "Validation Distance (km)": val_metric,
                    "Validation Accuracy Top 1": val_top1_accuracy,
                    "Validation Accuracy Top 3": val_top3_accuracy,
                    "Validation Accuracy Top 5": val_top5_accuracy,
                    "Validation Accuracy Top 1 Country": val_top1_correct_country,
                    "Validation Accuracy Top 3 Country": val_top3_correct_country,
                    "Validation Accuracy Top 5 Country": val_top5_correct_country
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
          run_dir = wandb.run.dir
          # Get directory of the run (without /files)
          run_dir = os.path.dirname(run_dir) if run_dir.endswith("files") else run_dir
          print('Saving artifacts to:', run_dir)
          wandb_model_path = os.path.join(wandb.run.dir, raw_model_path)
          
          if self.country_to_index is not None:
              # copy to run directory
              wandb_country_to_index_file = os.path.join(run_dir, 'country_to_index.json')
              # write json file
              with open(wandb_country_to_index_file, 'w') as f:
                  json.dump(self.country_to_index, f)
              # save to wandb
              wandb.save(wandb_country_to_index_file)
              
          if self.region_to_index is not None:
              # copy to run directory
              wandb_region_to_index_file = os.path.join(run_dir, 'region_to_index.json')
              # write json file
              with open(wandb_region_to_index_file, 'w') as f:
                  json.dump(self.region_to_index, f)
              # save to wandb
              wandb.save(wandb_region_to_index_file)
              
          if self.region_index_to_middle_point is not None:
              # copy to run directory
              wandb_region_index_to_middle_point_file = os.path.join(run_dir, 'region_index_to_middle_point.json')
              # write json file
              with open(wandb_region_index_to_middle_point_file, 'w') as f:
                  json.dump(self.region_index_to_middle_point, f)
              # save to wandb
              wandb.save(wandb_region_index_to_middle_point_file)
              
          if self.region_index_to_country_index is not None:
              # copy to run directory
              wandb_region_index_to_country_index_file = os.path.join(run_dir, 'region_index_to_country_index.json')
              # write json file
              with open(wandb_region_index_to_country_index_file, 'w') as f:
                  json.dump(self.region_index_to_country_index, f)
              # save to wandb
              wandb.save(wandb_region_index_to_country_index_file)
              
          if self.test_data_path is not None:
            # Copy test data to run directory
            wandb_test_data_path = os.path.join(run_dir, 'test_data.pth')
            # write json file
            shutil.copy(self.test_data_path, wandb_test_data_path)
            wandb.save(wandb_test_data_path)
            # Only save the test data once
            self.test_data_path = None
              
          torch.save(best_model.state_dict(), wandb_model_path)
          wandb.save(wandb_model_path)
          
          # Clean up
          del self.model
          del best_model

          gc.collect()
          torch.cuda.empty_cache()  

  def run_epoch(self, optimizer, is_train=True, is_test=False, use_balanced_accuracy=False):
    if is_train:
        self.model.train()
    else:
        self.model.eval()

    total_loss = 0.0
    total_metric = 0.0
    top1_correct = 0
    top1_correct_country = 0
    top3_correct = 0
    top3_correct_country = 0
    top5_correct = 0
    top5_correct_country = 0
    # for balanced accuracy calculation
    all_targets = []
    all_predictions = []
    # for country accuracy when predicting regions
    all_country_targets = []
    all_country_predictions = []

    data_loader = self.train_dataloader if is_train else (self.val_dataloader if not is_test else self.test_dataloader)
    
    # Because now the same country can be in the top 5 multiple times, we need to calculate the top k correct differently
    # Only needed for the country accuracy when predicting regions
    # Do not use for the normal country/region accuracy because the top 5 predictions are unique and this is less efficient
    def calculate_topk_multiple(countries_correct, k):
        return torch.min(countries_correct[:, :k].sum(dim=1), torch.ones(countries_correct.size(0), device=self.device)).sum().item()

    for images, coordinates, country_indices, region_indices in data_loader:
        with torch.set_grad_enabled(is_train):
            images = images.to(self.device)
            targets = coordinates.to(self.device) if self.use_coordinates else (region_indices.to(self.device) if self.use_regions else country_indices.to(self.device))
            if is_train:
                optimizer.zero_grad()
            outputs = self.model(images)
            probabilities = F.softmax(outputs, dim=1)
            loss = self.criterion(outputs, targets) if not self.use_regions else self.criterion(outputs, self.region_middle_points, coordinates)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)

            if self.use_coordinates:
                total_metric += self.mean_spherical_distance(outputs, targets).item() * images.size(0)
            if self.use_regions:
                total_metric += self.mean_spherical_distance(self.region_middle_points[outputs.argmax(dim=1)], self.region_middle_points[targets]).item() * images.size(0)
            if not self.use_coordinates:
                # Get the top 5 predictions for each image
                _, predicted_top5 = probabilities.topk(5, 1, True, True)

                # Calculate different accuracies
                correct = predicted_top5.eq(targets.view(-1, 1).expand_as(predicted_top5))

                top1_correct += correct[:, 0].sum().item()
                top3_correct += correct[:, :3].sum().item()
                top5_correct += correct[:, :5].sum().item()
                
                # Store all targets and predictions for balanced accuracy calculation
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted_top5[:, 0].cpu().numpy())

                if self.use_regions:
                    # Get the country for each region
                    target_countries = country_indices.to(self.device)
                    predicted_countries_top5 = torch.tensor([[self.region_index_to_country_index.get(region_index.item(), -1) for region_index in top5] for top5 in predicted_top5]).to(self.device)
                    countries_correct = predicted_countries_top5.eq(target_countries.view(-1, 1).expand_as(predicted_countries_top5))
                    
                    # Calculate different accuracies
                    top1_correct_country += calculate_topk_multiple(countries_correct, 1)
                    top3_correct_country += calculate_topk_multiple(countries_correct, 3)
                    top5_correct_country += calculate_topk_multiple(countries_correct, 5)
                    
                    # Store all country targets and predictions for balanced accuracy calculation
                    all_country_targets.extend(target_countries.cpu().numpy())
                    all_country_predictions.extend(predicted_countries_top5[:, 0].cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    
    if self.use_coordinates or self.use_regions:
        avg_metric = total_metric / len(data_loader.dataset)

    if not self.use_coordinates:
        top1_accuracy = top1_correct / len(data_loader.dataset)
        top3_accuracy = top3_correct / len(data_loader.dataset)
        top5_accuracy = top5_correct / len(data_loader.dataset)

        if use_balanced_accuracy:
            balanced_acc = balanced_accuracy_score(all_targets, all_predictions)
        else:
            balanced_acc = None

    if self.use_coordinates:
        return avg_loss, avg_metric
    elif self.use_regions:
        top1_correct_country = top1_correct_country / len(data_loader.dataset)
        top3_correct_country = top3_correct_country / len(data_loader.dataset)
        top5_correct_country = top5_correct_country / len(data_loader.dataset)
        
        if use_balanced_accuracy:
            balanced_country_acc = balanced_accuracy_score(all_country_targets, all_country_predictions)
        else:
            balanced_country_acc = None
            
        if balanced_acc is not None and balanced_country_acc is not None:
            return avg_loss, avg_metric, top1_accuracy, top3_accuracy, top5_accuracy, top1_correct_country, top3_correct_country, top5_correct_country, balanced_acc, balanced_country_acc
        return avg_loss, avg_metric, top1_accuracy, top3_accuracy, top5_accuracy, top1_correct_country, top3_correct_country, top5_correct_country
    else:
        if balanced_acc is not None:
            return avg_loss, top1_accuracy, top3_accuracy, top5_accuracy, balanced_acc
        return avg_loss, top1_accuracy, top3_accuracy, top5_accuracy
