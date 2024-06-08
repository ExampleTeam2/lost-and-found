import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F



class GeolocalizationLoss(nn.Module):
    def __init__(self, centroids, temperature=75):
        super(GeolocalizationLoss, self).__init__()
        self.temperature = temperature
        self.centroids = centroids
        self.rad_torch = torch.tensor(6378137.0, dtype=torch.float64)

    def haversine_distance(self, p1, p2):
        # Haversine formula to compute the distance between two points (in radians)
        r = 6371  # Earth's radius in kilometers
        lambda1, phi1 = p1[:, 0], p1[:, 1]
        lambda2, phi2 = p2[:, 0], p2[:, 1]

        delta_phi = phi2 - phi1
        delta_lambda = lambda2 - lambda1

        a = torch.sin(delta_phi / 2) ** 2 + torch.cos(phi1) * torch.cos(phi2) * torch.sin(delta_lambda / 2) ** 2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
        distance = r * c
        return distance

    def haversine_smooth(self, true_coords, true_geocell, predicted_geocells):
        # Calculate smoothed labels based on the Haversine distance
        true_geocell_centroid = self.centroids[true_geocell]
        true_distances = self.haversine_distance(true_coords, true_geocell_centroid.unsqueeze(0))

        smoothed_labels = []
        for i, centroid in enumerate(self.centroids):
            centroid_distance = self.haversine_distance(true_coords, centroid.unsqueeze(0))
            smoothed_label = torch.exp(-(centroid_distance - true_distances) / self.temperature)
            smoothed_labels.append(smoothed_label)
        smoothed_labels = torch.stack(smoothed_labels).squeeze()

        return smoothed_labels

    def haversine_matrix(self, x,y):
      """Computes the haversine distance between two matrices of points

      Args:
          x (Tensor): matrix 1 (lon, lat) -> shape (N, 2)
          y (Tensor): matrix 2 (lon, lat) -> shape (2, M)

      Returns:
          Tensor: haversine distance in km -> shape (N, M)
      """
      x_rad, y_rad = torch.deg2rad(x), torch.deg2rad(y)
      delta = x_rad.unsqueeze(2) - y_rad
      p = torch.cos(x_rad[:, 1]).unsqueeze(1) * torch.cos(y_rad[1, :]).unsqueeze(0)
      a = torch.sin(delta[:, 1, :] / 2)**2 + p * torch.sin(delta[:, 0, :] / 2)**2
      c = 2 * torch.arcsin(torch.sqrt(a))
      km = (self.rad_torch * c) / 1000
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

    def forward(self, outputs, true_coords, true_geocell):
        
        # Calculate the smoothed labels
        smoothed_labels = self.haversine_smooth(true_coords, true_geocell, self.centroids)

        # Calculate the cross-entropy loss using the smoothed labels
        log_probs = F.log_softmax(outputs, dim=1)
        loss = -torch.sum(smoothed_labels * log_probs, dim=1).mean()
        
        
        return loss
