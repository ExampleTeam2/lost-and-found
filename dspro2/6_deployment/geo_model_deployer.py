import torch
import sys

sys.path.insert(0, '../4_modeling')
from coordinate_handler import cartesian_to_coordinates
sys.path.insert(0, '../5_evaluation')
from geo_model_inference import GeoModelInference

class GeoModelDeployer(GeoModelInference):
  def __init__(self, num_classes=3, predict_coordinates=False, country_to_index=None, region_to_index=None, region_index_to_middle_point=None, region_index_to_country_index=None, predict_regions=False):
      super().__init__(num_classes=num_classes, predict_coordinates=predict_coordinates, country_to_index=country_to_index, region_to_index=region_to_index, region_index_to_middle_point=region_index_to_middle_point, region_index_to_country_index=region_index_to_country_index, predict_regions=predict_regions)

  # When predicting regions, countries are in order of summed probability and not in order of the probabilities of the corresponding regions.
  # This means that the top region can not be within the top country. For the countries corresponding to the regions, use corresponding_countries.
  def predict(self, data_loader):
    self.model.eval()
    with torch.no_grad():
      for images in data_loader:
        if not self.use_coordinates:
          probabilities, outputs = self.forward(images)
        else:
          outputs = self.forward(images)
          
    if self.predict_coordinates:
      coordinates = [cartesian_to_coordinates(*(coordinate.cpu().numpy())) for coordinate in outputs]
      return coordinates, outputs.cpu().item()
    
    else:
      probabilities_sorted, indices_sorted = probabilities.topk(probabilities.size()[-1], 1, True, True)
      index_to_name = self.index_to_country if not self.predict_regions else self.index_to_region
      names = list([[index_to_name.get(item.cpu().item(), None) for item in indices] for indices in indices_sorted])
      if not self.predict_regions:
        return names, indices_sorted.cpu().item(), probabilities_sorted.cpu().item()
      else:
        corresponding_countries = []
        countries = []
        country_indices = []
        country_probabilities = []
        
        for indices, probabilities in zip(indices_sorted, probabilities_sorted):
          current_countries = []
          current_country_indices_set = set()
          current_country_indices = []
          current_country_probabilities = []
          
          for region_index, probability in zip(indices, probabilities):
            probability = probability.cpu().item()
            
            country_index = self.region_index_to_country_index.get(region_index.cpu().item(), -1)
            
            if (country_index >= 0 and country_index not in current_country_indices_set):
              country_name = self.index_to_country.get(country_index, None)
              current_countries.append(country_name)
              current_country_indices.append(country_index)
              current_country_indices_set.add(country_index)
              current_country_probabilities.append(probability)
            else:
              current_country_probabilities[-1] += probability
            
          corresponding_countries.append(current_countries.copy())
          # Sort countries by summed probability
          current_countries, current_country_indices, current_country_probabilities = zip(*sorted(zip(current_countries, current_country_indices, current_country_probabilities), key=lambda x: x[2], reverse=True))
          countries.append(current_countries)
          country_indices.append(current_country_indices)
          country_probabilities.append(current_country_probabilities)
          
        return names, indices_sorted.cpu().item(), probabilities_sorted.cpu().item(), countries, country_indices, country_probabilities, corresponding_countries
