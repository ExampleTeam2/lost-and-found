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
        lambda1, phi1 = torch.deg2rad(p1[:, 0]), torch.deg2rad(p1[:, 1])
        lambda2, phi2 = torch.deg2rad(p2[:, 0]), torch.deg2rad(p2[:, 1])

        delta_phi = phi2 - phi1
        delta_lambda = lambda2 - lambda1

        a = torch.sin(delta_phi / 2) ** 2 + torch.cos(phi1) * torch.cos(phi2) * torch.sin(delta_lambda / 2) ** 2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
        distance = r * c
        return distance

    def haversine_smooth(self, true_coords, true_geocell):
        batch_size = true_coords.size(0)
        num_geocells = self.centroids.size(0)
        
        true_geocell_centroid = self.centroids[true_geocell]  # Shape: [batch_size, 2]
        true_coords_expanded = true_coords.unsqueeze(1).expand(-1, num_geocells, -1)  # Shape: [batch_size, num_geocells, 2]
        true_geocell_centroid_expanded = true_geocell_centroid.unsqueeze(1).expand(-1, num_geocells, -1)  # Shape: [batch_size, num_geocells, 2]

        true_distances = self.haversine_distance(true_coords_expanded.reshape(-1, 2), true_geocell_centroid_expanded.reshape(-1, 2))  # Shape: [batch_size * num_geocells]
        true_distances = true_distances.reshape(batch_size, num_geocells)  # Shape: [batch_size, num_geocells]
        
        smoothed_labels = []
        for i in range(num_geocells):
            centroid = self.centroids[i].unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, 2]
            centroid_distance = self.haversine_distance(true_coords_expanded[:, i, :], centroid)
            smoothed_label = torch.exp(-(centroid_distance - true_distances[:, i]) / self.temperature)
            smoothed_labels.append(smoothed_label.unsqueeze(1))
        smoothed_labels = torch.cat(smoothed_labels, dim=1)  # Shape: [batch_size, num_geocells]

        return smoothed_labels


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
        self.centroids = self.centroids.to(outputs.device)
        true_coords = true_coords.to(outputs.device)
        true_geocell = true_geocell.to(outputs.device)
        
        # Calculate the smoothed labels
        smoothed_labels = self.haversine_smooth(true_coords, true_geocell)

        # Calculate the cross-entropy loss using the smoothed labels
        log_probs = F.log_softmax(outputs, dim=1)
        loss = -torch.sum(smoothed_labels * log_probs, dim=1).mean()
        
        
        return loss
