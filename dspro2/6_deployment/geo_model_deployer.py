import torch
import sys
import numpy as np

sys.path.insert(0, '../4_modeling')
from coordinate_handler import cartesian_to_coordinates
sys.path.insert(0, '../5_evaluation')
from geo_model_inference import GeoModelInference

class GeoModelDeployer(GeoModelInference):
  def __init__(self, num_classes=3, predict_coordinates=False, country_to_index=None, region_to_index=None, region_index_to_middle_point=None, region_index_to_country_index=None, predict_regions=False):
      super().__init__(num_classes=num_classes, predict_coordinates=predict_coordinates, country_to_index=country_to_index, region_to_index=region_to_index, region_index_to_middle_point=region_index_to_middle_point, region_index_to_country_index=region_index_to_country_index, predict_regions=predict_regions)
      
  # When predicting regions, countries are in order of summed probability and not in order of the probabilities of the corresponding regions.
  # This means that the top region can not be within the top country. For the countries corresponding to the regions, use corresponding_countries.
  # For coordinate prediction they will be returned reversed from the training data: Lon, Lat instead of Lat, Lon.
  # Set top_n to 0 to get all countries and regions.
  def predict(self, data_loader, top_n=0):
    self.model.eval()
    with torch.no_grad():
      for images in data_loader:
        if not self.use_coordinates:
          probabilities, outputs = self.forward(images)
        else:
          projected, outputs = self.forward(images)
          
    if self.use_coordinates:
      # Use projected cartesian coordinates for more accuracy
      all_coordinates = [cartesian_to_coordinates(*(coordinate.cpu().numpy())) for coordinate in projected]
      # Reverse the order of the coordinates
      all_coordinates = [[(coordinate[1], coordinate[0]) for coordinate in coordinates] for coordinates in all_coordinates]
      return np.array(all_coordinates), outputs.cpu().numpy()
    
    else:
      probabilities_sorted, indices_sorted = probabilities.topk(probabilities.size()[-1], 1, True, True)
      original_probabilities_sorted = probabilities_sorted.clone()
      original_indices_sorted = indices_sorted.clone()
      actual_top_n = min(top_n, indices_sorted.size()[-1]) if top_n > 0 else indices_sorted.size()[-1]
      probabilities_sorted = probabilities_sorted[:, :actual_top_n]
      indices_sorted = indices_sorted[:, :actual_top_n]
      index_to_name = self.index_to_country if not self.use_regions else self.index_to_region
      names = list([[index_to_name.get(item.cpu().item(), None) for item in indices] for indices in indices_sorted])
      
      if not self.use_regions:
        return names, indices_sorted.cpu().numpy(), probabilities_sorted.cpu().numpy()
      else:
        corresponding_countries = []
        corresponding_country_indices = []
        countries = []
        country_indices = []
        country_probabilities = []
        
        for indices, probabilities in zip(original_indices_sorted, original_probabilities_sorted):
          current_corresponding_country_indices = []
          current_country_indices_set = set()
          current_country_indices = []
          current_country_probabilities = []
          
          for region_index, probability in zip(indices, probabilities):
            probability = probability.cpu().item()
            
            country_index = self.region_index_to_country_index.get(region_index.cpu().item(), -1)
            
            if country_index >= 0:
              if country_index not in current_country_indices_set:
                current_corresponding_country_indices.append(country_index)
                current_country_indices.append(country_index)
                current_country_indices_set.add(country_index)
                current_country_probabilities.append(probability)
              else:
                current_corresponding_country_indices.append(current_corresponding_country_indices[-1])
                current_country_probabilities[-1] += probability
                
          # Sort countries by summed probability
          current_country_indices, current_country_probabilities = zip(*list(sorted(zip(current_country_indices, current_country_probabilities), key=lambda x: x[1], reverse=True)))
              
          actual_top_n_countries = min(top_n, len(current_country_indices)) if top_n > 0 else len(current_country_indices)
          current_country_indices = current_country_indices[:actual_top_n_countries]
          current_country_probabilities = current_country_probabilities[:actual_top_n_countries]
          current_countries = [self.index_to_country.get(country_index, None) for country_index in current_country_indices]
          
          current_corresponding_country_indices = current_corresponding_country_indices[:actual_top_n]
          current_corresponding_countries = [self.index_to_country.get(country_index, None) for country_index in current_corresponding_country_indices]
          
          corresponding_countries.append(current_corresponding_countries)
          corresponding_country_indices.append(current_corresponding_country_indices)
          countries.append(current_countries)
          country_indices.append(current_country_indices)
          country_probabilities.append(current_country_probabilities)
          
        return names, indices_sorted.cpu().numpy(), probabilities_sorted.cpu().numpy(), countries, np.array(country_indices), np.array(country_probabilities), corresponding_countries, np.array(corresponding_country_indices)
      
  def predict_single(self, data_loader, top_n=0):
        if self.use_coordinates:
          coordinates, outputs = self.predict(data_loader, top_n)
          return coordinates[0], outputs[0]
        elif not self.use_regions:
          names, indices, probabilities = self.predict(data_loader, top_n)
          return names[0], indices[0], probabilities[0]
        else:
          names, indices, probabilities, countries, country_indices, country_probabilities, corresponding_countries, corresponding_country_indices = self.predict(data_loader, top_n)
          return names[0], indices[0], probabilities[0], countries[0], country_indices[0], country_probabilities[0], corresponding_countries[0], corresponding_country_indices[0]
