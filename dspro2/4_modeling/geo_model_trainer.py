import random
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
import numpy as np

import wandb
import uuid


class GeoModelTrainer:
  def __init__(self, datasize, train_dataloader, val_dataloader, num_classes=2, predict_coordinates=True):
      self.num_classes = num_classes
      self.train_dataloader = train_dataloader
      self.val_dataloader = val_dataloader
      self.datasize = datasize
      self.use_coordinates = predict_coordinates
      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
      self.model_type = None
      self.model = None

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
      else:
          raise ValueError("Unsupported model type. Supported types are: resnet18, resnet34, resnet50, resnet101, resnet152.")
      
      # Modify the final layer based on the number of classes
      model.fc = nn.Linear(model.fc.in_features, self.num_classes)
      nn.init.kaiming_normal_(model.fc.weight, mode='fan_out', nonlinearity='relu')
      nn.init.constant_(model.fc.bias, 0)
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
          patience = 20

          # Rename run name and initialize parameters in model name
          model_name = f"model_{config.model_name}_lr_{config.learning_rate}_opt_{config.optimizer}_weightDecay_{config.weight_decay}"
          run_name = model_name + f"_{uuid.uuid4()}"
          wandb.run.name = run_name

          # Initialize model, optimizer and criterion
          self.model = self.initialize_model(model_type=config.model_name).to(self.device)

          if self.use_coordinates:
            criterion = nn.MSELoss()
          else:
            criterion = nn.CrossEntropyLoss()

          optimizer_grouped_parameters = [
              {"params": [p for n, p in self.model.named_parameters() if not n.startswith('fc')], "lr": config.learning_rate * 0.1},
              {"params": self.model.fc.parameters(), "lr": config.learning_rate}
          ]
          optimizer = optim.AdamW(optimizer_grouped_parameters, weight_decay=config.weight_decay)

          for epoch in range(config.epochs):
              if self.use_coordinates:
                  train_loss, train_metric = self.run_epoch(criterion, optimizer, is_train=True)
                  val_loss, val_metric = self.run_epoch(criterion, optimizer, is_train=False)
              else:
                  train_loss, train_top1_accuracy, train_top3_accuracy, train_top5_accuracy = self.run_epoch(criterion, optimizer, is_train=True)
                  val_loss, val_top1_accuracy, val_top3_accuracy, val_top5_accuracy = self.run_epoch(criterion, optimizer, is_train=False)
              
              # Early stopping and logging
              if (self.use_coordinates and val_metric < best_val_metric) or (not self.use_coordinates and val_top1_accuracy > best_val_metric):
                  #if val_metric:
                  #    best_val_metric = val_metric
                  #else:
                  best_val_metric = val_top1_accuracy
                  torch.save(self.model.state_dict(), f"models/datasize_{self.datasize}/best_model_checkpoint{model_name}_predict_coordinates_{self.use_coordinates}.pth")
                  patience_counter = 0
              else:
                  patience_counter += 1
                  if patience_counter >= patience:
                      print(f"Stopping early after {patience} epochs without improvement")
                      break

              # Log metrics to wandb
              if self.use_coordinates:
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

    for images, coordinates, country_indices in data_loader:
        with torch.set_grad_enabled(is_train):
            images = images.to(self.device)
            targets = coordinates.to(self.device) if self.use_coordinates else country_indices.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, targets)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            if self.use_coordinates:
                total_metric += self.mean_spherical_distance(outputs, targets).item() * images.size(0)
            else:
                # Get the top 5 predictions for each image
                _, predicted_top5 = outputs.topk(5, 1, True, True)

                # Calculate different accuracies
                correct = predicted_top5.eq(targets.view(-1, 1).expand_as(predicted_top5))

                top1_correct += correct[:, 0].sum().item()
                top3_correct += correct[:, :3].sum().item()
                top5_correct += correct[:, :5].sum().item()

    avg_loss = total_loss / len(data_loader.dataset)

    if self.use_coordinates:
        avg_metric = total_metric / len(data_loader.dataset)
        return avg_loss, avg_metric
    else:
        top1_accuracy = top1_correct / len(data_loader.dataset)
        top3_accuracy = top3_correct / len(data_loader.dataset)
        top5_accuracy = top5_correct / len(data_loader.dataset)
        return avg_loss, top1_accuracy, top3_accuracy, top5_accuracy
