import os
import re
import json
import pycountry
import reverse_geocoder as rg
from fuzzywuzzy import process

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

class CountryEnricher:
    """Enriches JSON files with country information based on coordinates."""
    def __init__(self, input_dir, output_dir, from_country=False):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.json_files = {}
        self.coordinates = [] # Batching for faster results, 5 x times faster now
        self.raw_countries = []
        self.file_map = {} # Batching for faster results, 5 x times faster now
        self.mode = 'singleplayer' if not from_country else 'multiplayer'
        self.pattern = r'singleplayer_(.+?)\.json' if not from_country else r'multiplayer_(.+?)\.json'
        self.from_country = from_country
        self.process()
    
    def load_and_prepare_files(self):
        json_paths = [f for f in os.listdir(self.input_dir) if f.endswith('.json') and self.mode in f]
        for file in json_paths:
            match = re.search(self.pattern, file)
            if match:
                image_id = match.group(1)
                try:
                  with open(os.path.join(self.input_dir, file), 'r') as f:
                    try:
                      json_data = json.load(f)
                    except json.JSONDecodeError:
                      print(f"JSONDecodeError in {file}")
                      print(f"JSON Data before Error-Handling: {json_data}")
                      f.seek(0)
                      json_data = f.read()
                      print(f"JSON Data after Error-Handling: {json_data}")
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
                except json.JSONDecodeError:
                  print(f"JSONDecodeError in {file}")
                  print(f"JSON Data: {json_data}")
    
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
    
    def save_enriched_files(self):
        os.makedirs(self.output_dir, exist_ok=True)
        for image_id, data in self.json_files.items():
            file_name = f"geoguessr_result_{self.mode}_" + image_id + ".json"
            file_path = os.path.join(self.output_dir, file_name)
            with open(file_path, 'w') as f:
                json.dump(data, f, ensure_ascii=False)
    
    def process(self):
        self.load_and_prepare_files()
        self.enrich_with_country_info()
        self.save_enriched_files()
