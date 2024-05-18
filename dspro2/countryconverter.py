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

def convert_from_coordinates(coordinates, file_map, json_files):
  results = rg.search(coordinates) if len(coordinates) else []
  for result, coord in zip(results, coordinates):
      image_ids = file_map[str(coord)]
      country_code = result['cc']
      country = pycountry.countries.get(alpha_2=country_code)
      if country is None:
        country_name = additional_countries_and_regions_from_country_code.get(country_code, None)
        if country_name is not None:
          country = AdditionalCountryOrRegion(country_code, country_name)
      if country is None:
        raise ValueError(f"Country code {country_code} not found")
      for image_id in image_ids:
        json_files[image_id]['country_name'] = country.name
        json_files[image_id]['country_code'] = country.alpha_2
  return json_files

def convert_country_to_correct_name(country_name):
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
  return country
  

def convert_from_names(file_map, json_files):
  for country_name, image_ids in file_map.items():
    country = convert_country_to_correct_name(country_name)
    
    for image_id in image_ids:
      json_files[image_id]['country_name'] = country.name
      json_files[image_id]['country_code'] = country.alpha_2
  return json_files
