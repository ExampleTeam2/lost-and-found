import torch
import torch.nn as nn
import torch.nn.functional as F

class GeolocalizationLoss(nn.Module):
    def __init__(self, temperature=75):
        super(GeolocalizationLoss, self).__init__()
        self.temperature = temperature
        self.rad_torch = torch.tensor(6378137.0, dtype=torch.float64)

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        radius = 6371  # km
        dlat = torch.deg2rad(lat2 - lat1)
        dlon = torch.deg2rad(lon2 - lon1)
        a = torch.sin(dlat / 2) ** 2 + torch.cos(torch.deg2rad(lat1)) * torch.cos(torch.deg2rad(lat2)) * torch.sin(dlon / 2) ** 2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
        distance = radius * c
        return distance

    def haversine_matrix(self, x, y):
        """Computes the haversine distance between two matrices of points

        Args:
            x (Tensor): matrix 1 (N, 2) -> (lat, lon)
            y (Tensor): matrix 2 (M, 2) -> (lat, lon)

        Returns:
            Tensor: haversine distance in km -> shape (N, M)
        """
        x_rad, y_rad = torch.deg2rad(x), torch.deg2rad(y)
        delta = x_rad.unsqueeze(2) - y_rad
        p = torch.cos(x_rad[:, 1]).unsqueeze(1) * torch.cos(y_rad[1, :]).unsqueeze(0)
        a = torch.sin(delta[:, 1, :] / 2)**2 + p * torch.sin(delta[:, 0, :] / 2)**2
        c = 2 * torch.arcsin(torch.sqrt(a))
        km = (rad_torch * c) / 1000
        return km

    def smooth_labels(self, distances):
        """Haversine smooths labels for shared representation learning across geocells.

        Args:
            distances (Tensor): distance (km) matrix of size (batch_size, num_geocells)

        Returns:
            Tensor: smoothed labels
        """
        adj_distances = distances - distances.min(dim=-1, keepdim=True)[0]
        smoothed_labels = torch.exp(-adj_distances / self.temperature)
        smoothed_labels = torch.nan_to_num(smoothed_labels, nan=0.0, posinf=0.0, neginf=0.0)
        return smoothed_labels

    def forward(self, outputs, targets, geocell_centroids, true_coords):
        batch_size, num_classes = outputs.size()
        device = outputs.device

        true_coords = true_coords.to(device)
        true_geocell_centroids = geocell_centroids[targets]

        print(f"True coords: {true_coords.shape}, True geocell centroids: {true_geocell_centroids.shape}, Geocell centroids: {geocell_centroids.T.shape}")

        # Compute the haversine distance between the true coordinates and the transformed geocell centroids
        haversine_distances = self.haversine_matrix(true_coords, geocell_centroids.T)

        # Smooth the labels
        smoothed_labels = self.smooth_labels(haversine_distances)

        # print shapes
        print(f"Smoothed labels: {smoothed_labels.shape}, Outputs: {outputs.shape}, Targets: {targets.shape}, True coords: {true_coords.shape}, Haversine distances: {haversine_distances.shape}")

        # Compute the cross-entropy loss
        loss = F.cross_entropy(outputs, targets, weight=smoothed_labels, reduction='mean')
        
        return loss
