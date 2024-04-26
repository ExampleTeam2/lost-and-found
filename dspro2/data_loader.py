import os
import json
import concurrent
import math
import random
import urllib3

def _get_counterpart(file):
  # Get the counterpart of a file (json or png)
  # Get the counterpart
  if file.endswith('.json'):
    counterpart = file.replace('result', 'location').replace('.json', '.png')
  elif file.endswith('.png'):
    counterpart = file.replace('location', 'result').replace('.png', '.json')
  else:
    raise ValueError('Invalid file type')
  return counterpart
    
# Get rid of unpaired files (where only either json or png is present)
def _remove_unpaired_files(files):
  paired_files = []
  for file in files:
    counterpart = _get_counterpart(file)
    if counterpart in files:
      paired_files.append(file)
  return paired_files

def get_countries_occurrences_from_files(files):
  # filter out-non json files
  multiplayer = list(filter(lambda x: x.endswith('.json'), files))
  # load all multiplayer data
  multiplayer_data = load_json_files(multiplayer)
  # get all countries with their number of games
  countries = {}
  countries_to_files = {}
  files_to_countries = {}
  for file, game in zip(multiplayer, multiplayer_data):
      country = game['country_name']
      if country in countries:
          countries[country] += 1
          countries_to_files[country].append(file)
      else:
          countries[country] = 1
          countries_to_files[country] = [file]
      files_to_countries[file] = country
          
          
  # sort countries by number of games
  sorted_countries = sorted(countries.items(), key=lambda x: x[1], reverse=True)
  # Update the dict to keep the order
  countries = dict(sorted_countries)
  return countries, countries_to_files, files_to_countries, len(multiplayer_data) 

# Assuming either just json or png is give, also return the others (in a flat list all together)
def _get_files_counterparts(files, all_files):
  files_counterparts = []
  for file in files:
    counterpart = _get_counterpart(file)
    if counterpart in all_files:
      files_counterparts.append(counterpart)
      files_counterparts.append(file)
    else:
      files_counterparts.append(file)
  return files_counterparts

# Replace one start of the file with another and create a map from the original files to the new ones, then return the new files and the map
def _convert_to_fake_locations(files, start, new_start):
  fake_files = [file.replace(start, new_start) for file in files]
  fake_locations_map = {new_file: file for file, new_file in zip(files, fake_files)}
  return fake_files, fake_locations_map

# From a list of files and a map, restore the original locations
def _restore_from_fake_locations(files, fake_locations_map):
  return [fake_locations_map.get(file, file) for file in files]

# Takes in a list of files and a occurrence map (from a different_dataset)), create an optimally mapped list of files where the occurrences correspond to the map (or are multiples of them)
def map_occurrences_to_files(files, occurrence_map, allow_missing=False):
  # get the occurrences of the files itself
  files_occurrences, countries_to_files, _, _ = get_countries_occurrences_from_files(files)
  # get the factors between each of the countries (nan if not in the map)
  all_countries = [*occurrence_map.keys(), *files_occurrences.keys()]
  factors = [(occurrence_map[country] / files_occurrences[country]) if (country in occurrence_map and country in files_occurrences) else float('nan') for country in all_countries]
  # if any of the factors is nan, raise an exception
  if any([x != x for x in factors]):
    if allow_missing:
      # filter out the missing countries
      factors = [x for x in factors if x == x]
    else:
      #raise ValueError('Missing country in one of the maps')
      print('Missing country in one of the maps')
  if allow_missing and len(factors) == 0:
    #raise ValueError('No countries in commmon between the maps')
    print('No countries in commmon between the maps')
  # Get the lowest factor
  factor = min(factors)
  # Get the number of files to load by country
  new_occurrences = {country: math.ceil(occurrence_map[country] * factor) for country in occurrence_map}
  # Get the files to load
  files_to_load = []
  for country in new_occurrences:
    if allow_missing and country not in countries_to_files:
      continue
    files_to_load.extend(countries_to_files[country][:new_occurrences[country]])
  # Get the pairs of files to load
  files_counterparts = _get_files_counterparts(files_to_load, [*files, *countries_to_files.values()])
  return files_counterparts, len(files_to_load)

def _get_list_from_local_dir(file_location, json_file_location = None, image_file_location = None, filterText='singleplayer', type=''):
  all_locations = []
  if file_location is not None:
    all_locations.append([file_location, filterText, type])
  if json_file_location is not None and type != 'png':
    all_locations.append([json_file_location, filterText, 'json'])
  if image_file_location is not None and type != 'json':
    all_locations.append([image_file_location, filterText, 'png'])
  all_files = []
  fake_locations_map = {}
  for location, current_filter, current_type in all_locations:
    current_files = list([file.path for file in os.scandir(location)])
    if filterText:
      current_files = list(filter(lambda x: current_filter in x and x.endswith(current_type), current_files))
    if file_location != location and type == '' and file_location is not None:
      current_files, current_fake_locations_map = _convert_to_fake_locations(current_files, location, file_location)
      # extend fake_locations_maps with current_fake_locations_map
      fake_locations_map.update(current_fake_locations_map)
    all_files.extend(current_files)
  if not type:
    all_files = _remove_unpaired_files(all_files)
  return all_files, fake_locations_map

def _get_list_from_html_file(download_link):
  # get list of files from nginx or apache html file
  http = urllib3.PoolManager()
  response = http.request('GET', download_link)
  html = response.data.decode('utf-8')
  files = []
  for line in html.split('\n'):
    if 'href=' in line:
      file = line.split('href="')[1].split('"')[0]
      file_name = file.split('/')[-1] if '/' in file else file
      if file_name:
        files.append(file_name)
  return files

def _get_list_from_download_link(download_link, filterText='singleplayer', type=''):
  full_list = _get_list_from_html_file(download_link)
  all_files = [file for file in full_list if filterText in file and file.endswith(type)]
  if not type:
    all_files = _remove_unpaired_files(all_files)
  return all_files

def _get_download_link_from_for_files(download_link, files_to_download):
  return [[download_link + '/' + file, file] for file in files_to_download]

def _download_files_direct(download_link, files_to_download, current_location, num_connections=16):
  actual_download_links_and_files = _get_download_link_from_for_files(download_link, files_to_download)
  with concurrent.futures.ThreadPoolExecutor(max_workers=num_connections) as executor:
    list(executor.map(lambda x: urllib3.request.urlretrieve(x[0], os.path.join(current_location, x[1])), actual_download_links_and_files))

def _download_files(download_link, files_to_download, file_location, json_file_location = None, image_file_location = None):
  files_to_normal_location = []
  files_to_json_location = []
  files_to_image_location = []
  if json_file_location is not None:
    files_to_json_location = [file for file in files_to_download if file.endswith('.json')]
  else:
    files_to_normal_location = [file for file in files_to_download if file.endswith('.json')]
  if image_file_location is not None:
    files_to_image_location = [file for file in files_to_download if file.endswith('.png')]
  else:
    files_to_normal_location = [file for file in files_to_download if file.endswith('.png')]
  if len(files_to_normal_location):
    _download_files_direct(download_link, files_to_normal_location, file_location)
  if len(files_to_json_location):
    _download_files_direct(download_link, files_to_json_location, json_file_location)
  if len(files_to_image_location):
    _download_files_direct(download_link, files_to_image_location, image_file_location)
  pass

def _get_files_and_ensure_download(download_link, file_location, json_file_location = None, image_file_location = None, filterText='singleplayer', type=''):
  all_files = _get_list_from_download_link(download_link)
  local_files, _ = _get_list_from_local_dir(file_location, json_file_location, image_file_location, filterText, type)
  local_basename_files = set([os.path.basename(file) for file in local_files])
  # Check if file with same basename is already in the file_location, otherwise download
  files_to_download = []
  for file in all_files:
    if os.path.basename(file) in local_basename_files:
      continue
    # Download the file
    files_to_download.append(file)
  _download_files(download_link, files_to_download, file_location, json_file_location, image_file_location)
  return _get_list_from_local_dir(file_location, json_file_location, image_file_location, filterText, type)

# Get file paths of data to load, using multiple locations and optionally a map.
# If ran for a second time, it will use the previous files and otherwise error.
# The limit will automatically be shuffled (but returned in same order).
# If no shuffle seed is given they will be returned in the original order.
# If a download link is uses, it will be used instead of the file location and the files will be downloaded to the file location.
# Set the download link to "env" to use the environment variable "DOWNLOAD_LINK" as the download link.
def get_data_to_load(loading_file = './data_list', file_location = os.path.join(os.path.dirname(__file__), '1_data_collection/.data'), json_file_location = None, image_file_location = None, filterText='singleplayer', type='', limit=0, allow_new_file_creation=True, countries_map=None, allow_missing_in_map=False, passthrough_map=False, shuffle_seed=None, download_link=None, return_basenames_too=False):
  if download_link == 'env':
    download_link = os.environ.get('DOWNLOAD_LINK')
    if not download_link:
      download_link = None
  all_files, fake_locations_map = _get_list_from_local_dir(file_location, json_file_location, image_file_location, filterText, type) if download_link is None else _get_files_and_ensure_download(download_link, file_location, json_file_location, image_file_location, filterText, type)
  if countries_map and not passthrough_map:
    all_files, _ = map_occurrences_to_files(all_files, countries_map, allow_missing=allow_missing_in_map)
  if limit:
    random.seed(shuffle_seed if shuffle_seed is not None else 42)
    random_perm_files = random.sample(all_files, len(all_files))
    limited_files = random_perm_files[:limit]
    all_files = [file for file in all_files if file in limited_files]
  if shuffle_seed is not None:
    random.seed(shuffle_seed)
    random.shuffle(all_files)
    
  all_files = _restore_from_fake_locations(all_files, fake_locations_map)
    
  base_files = list([os.path.basename(file) for file in all_files])
  files_to_load = base_files
  
  try:
    if os.stat(loading_file):
      with open(loading_file, 'r', encoding='utf8') as file:
        files_to_load = file.read().split('\n')
  except FileNotFoundError:
    if not allow_new_file_creation:
      #raise ValueError('No loading file at location')
      print('No loading file at location')
    pass
      
  if not len(files_to_load):
    #raise ValueError('No files to load, did you forget to specify the type? Otherwise unpaired files will be ignored')
    print('No files to load, did you forget to specify the type? Otherwise unpaired files will be ignored')
  
  if not len(all_files):
    #raise ValueError('No files in loading location')
    print('No files in loading location')
  
  actual_file_locations = []
  
  for file in files_to_load:
    if file not in base_files:
      #raise ValueError('Missing file ' + file)
      print('Missing file ' + file)
    else:
      actual_file_locations.append(all_files[base_files.index(file)])
      
  with open(loading_file, 'w', encoding='utf8') as file:
    file.write('\n'.join(files_to_load))
    
  if return_basenames_too:
    return actual_file_locations, files_to_load
    
  return actual_file_locations

# Update data based on factors
def update_data_to_load(files_to_keep, old_loading_file = './data_list', new_loading_file = './updated_data_list', file_location = os.path.join(os.path.dirname(__file__), '1_data_collection/.data'), json_file_location = None, image_file_location = None, filterText='singleplayer', type='', limit=0):
  _, previous_files_to_load = get_data_to_load(old_loading_file, file_location, json_file_location, image_file_location, filterText, type, limit, allow_new_file_creation=False, return_basenames_too=True)
  files_to_load = []
  base_files_to_keep = list([os.path.basename(file) for file in files_to_keep])
  for previous_file_to_load in previous_files_to_load:
    if previous_file_to_load in base_files_to_keep:
      files_to_load.append(previous_file_to_load)
     
  base_files = files_to_load
      
  try:
    if os.stat(new_loading_file):
      with open(new_loading_file, 'r', encoding='utf8') as file:
        files_to_load = file.read().split('\n')
  except FileNotFoundError:
    pass
  
  for file in files_to_load:
    if file not in base_files:
      #raise ValueError('Missing file ' + file)
      print('Missing file ' + file)
      
  if not len(files_to_load):
    #raise ValueError('No files to load, did you forget to specify the type? Otherwise unpaired files will be ignored')
    print('No files to load')
    print(files_to_load)
      
  with open(new_loading_file, 'w', encoding='utf8') as file:
    file.write('\n'.join(files_to_load))


# load a single json file
def load_json_file(file):
  with open(file, 'r', encoding='utf8') as f:
    return json.load(f)

# load mutliple json files parallelized
def load_json_files(files, num_workers=16):
  with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    results = list(executor.map(load_json_file, files))
  return results   

# get countries occurrences from multiplayer games
def get_countries_occurrences(loading_file = './countries_map_data_list', file_location = os.path.join(os.path.dirname(__file__), '1_data_collection/.data'), filterText='multiplayer'):
  multiplayer = get_data_to_load(loading_file=loading_file, file_location=file_location, filterText=filterText, type='json')
  # map data
  countries, countries_to_files, files_to_countries, num_games = get_countries_occurrences_from_files(multiplayer)
  return countries, countries_to_files, files_to_countries, num_games
