import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, efficientnet_b1, efficientnet_b3, efficientnet_b4, efficientnet_b7
from torchvision.models import mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
from torchvision.models.efficientnet import EfficientNet_B1_Weights, EfficientNet_B3_Weights, EfficientNet_B4_Weights, EfficientNet_B7_Weights
import numpy as np
from sklearn.metrics import balanced_accuracy_score

from coordinate_handler import coordinates_to_cartesian, default_radius
from custom_haversine_loss import GeolocalizationLoss


class GeoModelHarness:
    def __init__(self, num_classes=3, predict_coordinates=False, country_to_index=None, region_to_index=None, region_index_to_middle_point=None, region_index_to_country_index=None, predict_regions=False):
        self.num_classes = num_classes
        self.use_coordinates = predict_coordinates
        self.use_regions = predict_regions
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model_type = None
        self.model = None
        self.country_to_index = country_to_index
        self.index_to_country = {v: k for k, v in country_to_index.items()} if country_to_index is not None else None
        self.region_to_index = region_to_index
        self.index_to_region = {v: k for k, v in region_to_index.items()} if region_to_index is not None else None
        self.region_index_to_middle_point = region_index_to_middle_point
        self.region_index_to_country_index = region_index_to_country_index

        # For inference/testing/training
        self.region_middle_points = np.array(list(self.region_index_to_middle_point.values()) if self.region_index_to_middle_point is not None else [], dtype="float64") if self.use_regions else None
        # Convert them to cartesian coordinates
        if self.region_middle_points is not None:
            self.region_middle_points = torch.tensor([coordinates_to_cartesian(*coordinate) for coordinate in self.region_middle_points], dtype=torch.float32).to(self.device)

        # For training/testing
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
        os.environ["PYTHONHASHSEED"] = str(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def initialize_model(self, model_type):
        self.model_type = model_type
        # Initialize already with best available weights (currently alias for IMAGENET1K_V2)
        if self.model_type == "resnet18":
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif self.model_type == "resnet34":
            model = resnet34(weights=ResNet34_Weights.DEFAULT)
        elif self.model_type == "resnet50":
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif self.model_type == "resnet101":
            model = resnet101(weights=ResNet101_Weights.DEFAULT)
        elif self.model_type == "resnet152":
            model = resnet152(weights=ResNet152_Weights.DEFAULT)
        elif self.model_type == "mobilenet_v2":
            model = mobilenet_v2(weights="IMAGENET1K_V2")
        elif self.model_type == "mobilenet_v3_small":
            model = mobilenet_v3_small(weights="IMAGENET1K_V1")
        elif self.model_type == "mobilenet_v3_large":
            model = mobilenet_v3_large(weights="IMAGENET1K_V1")
        elif self.model_type == "efficientnet_b1":
            model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
        elif self.model_type == "efficientnet_b3":
            model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        elif self.model_type == "efficientnet_b4":
            model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
        elif self.model_type == "efficientnet_b7":
            model = efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT)
        else:
            raise ValueError("Unsupported model type. Supported types are: resnet18, resnet34, resnet50, resnet101, resnet152, mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large, efficientnet_b1, efficientnet_b3, efficientnet_b4, efficientnet_b7")

        if "resnet" in self.model_type:
            # Modify the final layer based on the number of classes
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            # Initialize weights of the classifier
            nn.init.kaiming_normal_(model.fc.weight, mode="fan_out", nonlinearity="relu")
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
            model.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=False), nn.Linear(in_features=in_features, out_features=self.num_classes, bias=True))
            # Initialize weights of the classifier
            nn.init.kaiming_normal_(model.classifier[1].weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(model.classifier[1].bias, 0)
        elif "mobilenet_v3" in self.model_type:
            if self.model_type == "mobilenet_v3_small":
                in_features = 1024
            else:
                in_features = 1280
            # Modify the final layer based on the number of classes
            model.classifier[3] = torch.nn.Linear(in_features=in_features, out_features=self.num_classes, bias=True)
            # Initialize weights of the classifier
            nn.init.kaiming_normal_(model.classifier[3].weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(model.classifier[3].bias, 0)
        self.model = model.to(self.device)

    def project_cartesian(self, cartesian, R=default_radius):
        # Round the cartesian coordinates to the sphere (x^2 + y^2 + z^2 = R^2)
        # This is done by normalizing the vector and multiplying it by the radius
        norms = cartesian.norm(p=2, dim=-1, keepdim=True) + 1e-9  # Adding epsilon to avoid division by zero
        return (cartesian / norms) * R

    def spherical_distance(self, cartesian1, cartesian2, R=6371.0):
        dot_product = (cartesian1 * cartesian2).sum(dim=1)

        norms1 = cartesian1.norm(p=2, dim=1)
        norms2 = cartesian2.norm(p=2, dim=1)

        cos_theta = dot_product / (norms1 * norms2)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

        theta = torch.acos(cos_theta)
        # curved distance -> "Bogenmass"
        distance = R * theta
        return distance

    def forward(self, images):
        images = images.to(self.device)
        outputs = self.model(images)
        if not self.use_coordinates:
            probabilities = F.softmax(outputs, dim=1)
            return probabilities, outputs
        projected = self.project_cartesian(outputs)
        return projected, outputs

    # Because with regions the same country can be in the top 5 multiple times, we need to calculate the top k correct differently
    # Only needed for the country accuracy when predicting regions
    # Do not use for the normal country/region accuracy because the top 5 predictions are unique and this is less efficient
    def calculate_topk_multiple(self, countries_correct, k):
        return torch.min(countries_correct[:, :k].sum(dim=1), torch.ones(countries_correct.size(0), device=self.device)).sum().item()

    # Do not use for the normal country/region accuracy because this is not very efficient
    def calculate_accuracy_per_country(self, all_targets, all_predictions):
        # Convert all_targets and all_predictions to tensors if they aren't already
        all_targets = torch.tensor(all_targets)
        all_predictions = torch.tensor(all_predictions)

        # Get the unique countries in the targets
        unique_countries = all_targets.unique()

        accuracy_per_country = {}

        for country in unique_countries:
            if country != -1:
                # Get the indices where the targets are equal to the current country
                indices = all_targets == country

                # Calculate the number of correct predictions for this country
                correct_predictions = (all_predictions[indices] == country).sum().item()

                # Calculate the total number of instances for this country
                total_instances = indices.sum().item()

                # Calculate the accuracy for this country
                accuracy = correct_predictions / total_instances

                # Store the accuracy in the dictionary
                accuracy_per_country[self.index_to_country[country.item()]] = accuracy

        return accuracy_per_country

    def run_epoch(self, data_loader, is_train=False, use_balanced_accuracy=False, balanced_on_countries_only=None, accuracy_per_country=False, median_metric=False, optimizer=None):
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
        # for distance median calculation
        all_metrics = []

        # If balanced_on_countries_only is not None, convert the strings to indices
        if balanced_on_countries_only is not None:
            balanced_on_countries_only = [self.country_to_index[country] for country in balanced_on_countries_only]
            balanced_on_countries_only = set(balanced_on_countries_only)

        for images, coordinates, country_indices, region_indices in data_loader:
            with torch.set_grad_enabled(is_train):
                targets = coordinates.to(self.device) if self.use_coordinates else (region_indices.to(self.device) if self.use_regions else country_indices.to(self.device))
                if is_train:
                    optimizer.zero_grad()
                if not self.use_coordinates:
                    probabilities, outputs = self.forward(images)
                else:
                    projected, outputs = self.forward(images)
                loss = self.criterion(outputs, targets) if not self.use_regions else self.criterion(outputs, self.region_middle_points, coordinates)

                if is_train:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * images.size(0)

                if self.use_coordinates:
                    new_metrics = self.spherical_distance(projected, targets)
                    new_metric = new_metrics.mean().item()
                    if median_metric:
                        all_metrics.extend(new_metrics.cpu().numpy())
                    total_metric += new_metric * images.size(0)
                if self.use_regions:
                    new_metrics = self.spherical_distance(self.region_middle_points[outputs.argmax(dim=1)], self.region_middle_points[targets])
                    new_metric = new_metrics.mean().item()
                    if median_metric:
                        all_metrics.extend(new_metrics.cpu().numpy())
                    total_metric += new_metric * images.size(0)
                if not self.use_coordinates:
                    # Get the top 5 predictions for each image
                    _, predicted_top5 = probabilities.topk(5, 1, True, True)

                    # Calculate different accuracies
                    correct = predicted_top5.eq(targets.view(-1, 1).expand_as(predicted_top5))

                    top1_correct += correct[:, 0].sum().item()
                    top3_correct += correct[:, :3].sum().item()
                    top5_correct += correct[:, :5].sum().item()

                    # If balanced_on_countries_only is not None and we are not predicting regions, set all other countries to -1 in the targets
                    if balanced_on_countries_only is not None and not self.use_regions:
                        targets = torch.tensor([target if target in balanced_on_countries_only else -1 for target in targets.tolist()], device=targets.device)

                    # Store all targets and predictions for balanced accuracy calculation
                    all_targets.extend(targets.cpu().numpy())
                    all_predictions.extend(predicted_top5[:, 0].cpu().numpy())

                    if self.use_regions:
                        # Get the country for each region
                        target_countries = country_indices.to(self.device)

                        predicted_countries_top5 = torch.tensor([[self.region_index_to_country_index.get(region_index.item(), -1) for region_index in top5] for top5 in predicted_top5]).to(self.device)
                        countries_correct = predicted_countries_top5.eq(target_countries.view(-1, 1).expand_as(predicted_countries_top5))

                        # Calculate different accuracies
                        top1_correct_country += self.calculate_topk_multiple(countries_correct, 1)
                        top3_correct_country += self.calculate_topk_multiple(countries_correct, 3)
                        top5_correct_country += self.calculate_topk_multiple(countries_correct, 5)

                        # If balanced_on_countries_only is not None, set all other countries to -1 in the targets
                        if balanced_on_countries_only is not None:
                            target_countries = torch.tensor([target if target in balanced_on_countries_only else -1 for target in target_countries.tolist()], device=target_countries.device)

                        # Store all country targets and predictions for balanced accuracy calculation
                        all_country_targets.extend(target_countries.cpu().numpy())
                        all_country_predictions.extend(predicted_countries_top5[:, 0].cpu().numpy())

        return_dict = {}

        avg_loss = total_loss / len(data_loader.dataset)
        return_dict["avg_loss"] = avg_loss

        if self.use_coordinates or self.use_regions:
            avg_metric = total_metric / len(data_loader.dataset)
            return_dict["avg_metric"] = avg_metric
            if median_metric:
                median_metric = np.median(np.array(all_metrics))
                return_dict["median_metric"] = median_metric

        if not self.use_coordinates:
            top1_accuracy = top1_correct / len(data_loader.dataset)
            top3_accuracy = top3_correct / len(data_loader.dataset)
            top5_accuracy = top5_correct / len(data_loader.dataset)
            return_dict["top1_accuracy"] = top1_accuracy
            return_dict["top3_accuracy"] = top3_accuracy
            return_dict["top5_accuracy"] = top5_accuracy

            if use_balanced_accuracy:
                top1_balanced_accuracy = balanced_accuracy_score(all_targets, all_predictions)
                return_dict["top1_balanced_accuracy"] = top1_balanced_accuracy

            if accuracy_per_country and not self.use_regions:
                # Calculate the accuracy per country
                accuracy_per_country = self.calculate_accuracy_per_country(all_targets, all_predictions)
                return_dict["accuracy_per_country"] = accuracy_per_country

        if self.use_regions:
            top1_correct_country = top1_correct_country / len(data_loader.dataset)
            top3_correct_country = top3_correct_country / len(data_loader.dataset)
            top5_correct_country = top5_correct_country / len(data_loader.dataset)
            return_dict["top1_correct_country"] = top1_correct_country
            return_dict["top3_correct_country"] = top3_correct_country
            return_dict["top5_correct_country"] = top5_correct_country

            if use_balanced_accuracy:
                top1_balanced_accuracy_country = balanced_accuracy_score(all_country_targets, all_country_predictions)
                return_dict["top1_balanced_accuracy_country"] = top1_balanced_accuracy_country

            if accuracy_per_country:
                # Calculate the accuracy per country
                accuracy_per_country = self.calculate_accuracy_per_country(all_country_targets, all_country_predictions)
                return_dict["accuracy_per_country"] = accuracy_per_country

        return return_dict
