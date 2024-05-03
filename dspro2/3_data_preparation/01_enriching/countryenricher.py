import os
import re
import json
import pycountry
import concurrent
import reverse_geocoder as rg
from fuzzywuzzy import process
import sys
sys.path.insert(0, '../../')
from data_loader import resolve_env_variable, get_json_files, load_json_file

additional_countries_and_regions = [
  ['XK', 'Kosovo'],
]
additional_countries_and_regions_split = list(zip(*additional_countries_and_regions))
additional_countries_and_regions_split_reversed = additional_countries_and_regions_split[::-1] if len(additional_countries_and_regions_split) > 1 else additional_countries_and_regions_split
additional_countries_and_regions_reversed = list(zip(*additional_countries_and_regions_split_reversed))

additional_countries_and_regions_names = additional_countries_and_regions_split[1]

additional_countries_and_regions_from_name = dict(additional_countries_and_regions_reversed)

def get_closest_additional_country_or_region_from_name(name):
  found, score = process.extractOne(name, additional_countries_and_regions_names)
  if score <= 80:
    return None
  return found

additional_countries_and_regions_from_country_code = dict(additional_countries_and_regions)

class AdditionalCountryOrRegion:
  def __init__(self, country_code, country_name):
    self.alpha_2 = country_code
    self.name = country_name

# You can use the environment variables FILE_LOCATION and JSON_FILE_LOCATION to set the input and output directories
class CountryEnricher:
    """Enriches JSON files with country information based on coordinates."""
    def __init__(self, input_dir, output_dir, from_country=False, allow_env=False):
        super().__init__()
        self.input_dir = resolve_env_variable(input_dir, 'FILE_LOCATION', allow_env, 'JSON_FILE_LOCATION')
        self.output_dir = resolve_env_variable(output_dir, 'JSON_FILE_LOCATION', allow_env, 'FILE_LOCATION')
        self.json_files = {}
        self.coordinates = [] # Batching for faster results, 5 x times faster now
        self.raw_countries = []
        self.file_map = {} # Batching for faster results, 5 x times faster now
        self.mode = 'singleplayer' if not from_country else 'multiplayer'
        self.pattern = r'singleplayer_(.+?)\.json' if not from_country else r'multiplayer_(.+?)\.json'
        self.from_country = from_country
        self.process()
        
    def load_and_prepare_files(self, num_workers=16):
        json_paths = get_json_files(self.input_dir)
        had_error = False
        
        def process_json_file(file_path):
          global had_error
          file = os.path.basename(file_path)
          match = re.search(self.pattern, file)
          if match:
              image_id = match.group(1)
              try:
                json_data = load_json_file(file_path)
              except json.JSONDecodeError as error:
                had_error = True
                print(f"JSONDecodeError in {file}")
                print(error)
                return
              self.json_files[image_id] = json_data
              if not self.from_country:
                if 'coordinates' in json_data:
                  coord = tuple(json_data['coordinates'])
                  self.coordinates.append(coord)
                  self.file_map[coord] = image_id
                else:
                  raise ValueError(f"Coordinates not found in {image_id}")
              else:
                if 'country' in json_data:
                  country = json_data['country']
                  self.raw_countries.append(country)
                  if country in self.file_map:
                    self.file_map[country].append(image_id)
                  else:
                    self.file_map[country] = [image_id]
                else:
                  raise ValueError(f"Country not found in {image_id}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
          current_file = 0
          for _ in executor.map(process_json_file, json_paths):
            current_file += 1
            if (current_file and current_file % 1000 == 0) or current_file == len(json_paths):
              print('Enriched ' + str(current_file) + ' files')
            
                    
        if had_error:
          raise ValueError("JSONDecodeError in one or more files")
    
    def enrich_with_country_info(self):
      if not self.from_country:
        results = rg.search(self.coordinates)
        for result, coord in zip(results, self.coordinates):
            image_id = self.file_map[coord]
            country_code = result['cc']
            country = pycountry.countries.get(alpha_2=country_code)
            if country is None:
              country_name = additional_countries_and_regions_from_country_code.get(country_code, None)
              if country_name is not None:
                country = AdditionalCountryOrRegion(country_code, country_name)
            if country is None:
              raise ValueError(f"Country code {country_code} not found")
            self.json_files[image_id]['country_name'] = country.name
            self.json_files[image_id]['country_code'] = country.alpha_2
      else:
        for country_name, image_ids in self.file_map.items():
          countries = []
          try:
            countries = pycountry.countries.search_fuzzy(country_name)
          except LookupError:
            matched_name = get_closest_additional_country_or_region_from_name(country_name)
            if matched_name is not None:
              country_code = additional_countries_and_regions_from_name.get(matched_name, None)
              country = AdditionalCountryOrRegion(country_code, matched_name)
              countries.append(country)
          if not len(countries):
            raise ValueError(f"Country {country_name} not found")
          country = countries[0]
          for image_id in image_ids:
            self.json_files[image_id]['country_name'] = country.name
            self.json_files[image_id]['country_code'] = country.alpha_2
    
    def save_enriched_files(self, num_workers=16):
        os.makedirs(self.output_dir, exist_ok=True)
        
        def save_json_file(image_id, data):
          file_name = f"geoguessr_result_{self.mode}_" + image_id + ".json"
          file_path = os.path.join(self.output_dir, file_name)
          with open(file_path, 'w', encoding='utf8') as f:
              json.dump(data, f, ensure_ascii=False)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
          current_file = 0
          for _ in executor.map(save_json_file, self.json_files.keys(), self.json_files.values()):
            current_file += 1
            if (current_file and current_file % 1000 == 0) or current_file == len(self.json_files.keys()):
              print('Enriched ' + str(current_file) + ' files')
    
    def process(self):
        self.load_and_prepare_files()
        self.enrich_with_country_info()
        self.save_enriched_files()
