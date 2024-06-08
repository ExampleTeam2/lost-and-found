import torch
import torch.nn as nn
import torch.nn.functional as F

class GeolocalizationLoss(nn.Module):
    def __init__(self, temperature=75):
        super(GeolocalizationLoss, self).__init__()
        self.temperature = temperature

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        radius = 6371  # km
        dlat = torch.deg2rad(lat2 - lat1)
        dlon = torch.deg2rad(lon2 - lon1)
        a = torch.sin(dlat / 2) ** 2 + torch.cos(torch.deg2rad(lat1)) * torch.cos(torch.deg2rad(lat2)) * torch.sin(dlon / 2) ** 2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
        distance = radius * c
        return distance

    def forward(self, outputs, targets, geocell_centroids, true_coords):
        batch_size, num_classes = outputs.size()
        device = outputs.device

        #geocell_centroids = torch.tensor([(point.x, point.y) for point in geocell_centroids], dtype=torch.float64).to(device)
        true_coords = true_coords.to(device)

        true_geocell_centroids = geocell_centroids[targets]

        distances_to_geocells, distances_to_true_geocells = self.haversine_distance(
            true_coords[:, 0].unsqueeze(1), 
            true_coords[:, 1].unsqueeze(1), 
            geocell_centroids[:, 0].unsqueeze(0), 
            geocell_centroids[:, 1].unsqueeze(0)
        ), self.haversine_distance(
            true_coords[:, 0], 
            true_coords[:, 1], 
            true_geocell_centroids[:, 0], 
            true_geocell_centroids[:, 1]
        ).unsqueeze(1)


        # Stabilize the calculation
        distances_to_geocells = torch.clamp(distances_to_geocells, min=1e-6)
        distances_to_true_geocells = torch.clamp(distances_to_true_geocells, min=1e-6)
        

        targets_smoothed = torch.exp(-(distances_to_geocells - distances_to_true_geocells) / self.temperature)
        
        log_probs = F.log_softmax(outputs, dim=1)
        loss = -torch.sum(log_probs * targets_smoothed, dim=1).mean()
        
        return loss
