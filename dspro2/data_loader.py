import os
import json
import concurrent
import math
import random
import shutil
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
  print('Filtering out unpaired files')
  file_dict = {}
  paired_files = []

  # Create a dictionary to track each file and its counterpart's presence
  for file in files:
    counterpart = _get_counterpart(file)
    file_dict[file] = counterpart
    file_dict.setdefault(counterpart, None)

  # Collect files where both members of the pair are present
  paired_files = [file for file in files if file_dict[file_dict[file]] is not None]
  
  print('Filtered out ' + str(len(files) - len(paired_files)) + ' unpaired files')
  return paired_files

def get_countries_occurrences_from_files(files, basenames_to_locations_map=None):
  # filter out-non json files
  json_files = list(filter(lambda x: x.endswith('.json'), files))
  json_files_full_paths = json_files
  if basenames_to_locations_map:
    json_files_full_paths = _map_to_locations(json_files, basenames_to_locations_map)
  # load all multiplayer data
  json_data = load_json_files(json_files_full_paths)
  # get all countries with their number of games
  countries = {}
  countries_to_files = {}
  files_to_countries = {}
  for file, game in zip(json_files, json_data):
      if 'country_name' not in game and 'country' not in game:
        print('Country name not found in game: ' + file)
        continue
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
  return countries, countries_to_files, files_to_countries, len(json_data) 

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

# Replace one start of the file with another
def _get_basename(file):
  return os.path.basename(file)

# Replace one start of the file with another and create a map from the original files to the new ones, then return the new files and the map
def _map_to_basenames(files, basenames_to_locations_map={}):
  basenames = [_get_basename(file) for file in files]
  for file, basename in zip(files, basenames):
    if basename not in basenames_to_locations_map:
      basenames_to_locations_map[basename] = file
  return basenames, basenames_to_locations_map

# From a list of files and a map, restore the original locations
def _map_to_locations(files, basenames_to_locations_map):
  return [basenames_to_locations_map.get(file, file) for file in files]

# Takes in a list of files and a occurrence map (from a different_dataset)), create an optimally mapped list of files where the occurrences correspond to the map (or are multiples of them)
def map_occurrences_to_files(files, occurrence_map, allow_missing=False, basenames_to_locations_map=None):
  # get the occurrences of the files itself
  files_occurrences, countries_to_files, _, _ = get_countries_occurrences_from_files(files, basenames_to_locations_map=basenames_to_locations_map)
  # get the factors between each of the countries (nan if not in the map)
  all_countries = [*occurrence_map.keys(), *files_occurrences.keys()]
  factors = [(occurrence_map[country] / files_occurrences[country]) if (country in occurrence_map and country in files_occurrences) else float('nan') for country in all_countries]
  # if any of the factors is nan, raise an exception
  if any([x != x for x in factors]):
    if allow_missing:
      # filter out the missing countries
      factors = [x for x in factors if x == x]
    else:
      raise ValueError('Missing country in one of the maps')
  if allow_missing and len(factors) == 0:
    raise ValueError('No countries in commmon between the maps')
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

def _get_list_from_html_file(download_link):
  # get list of files from nginx or apache html file
  http = urllib3.PoolManager()
  print('Getting files list from remote')
  response = http.request('GET', download_link)
  html = response.data.decode('utf-8')
  print('Got files list from remote')
  files = []
  if not html:
    raise ValueError("No response from remote server")
  for line in html.split('<a'):
    if 'href=' in line:
      if not 'href="' in line and '"' in line[len('href='):]:
        raise ValueError("Invalid line in remote response: " + line)
      file = line.split('href="')[-1].split('"')[0]
      if not file:
        raise ValueError("Invalid link in remote response: " + line.split('href="')[-1])
      file_name = file.split('/')[-1] if '/' in file else file
      if file_name:
        files.append(file_name)
  files = [file for file in files if file is not None]
  print('Parsed files list from remote')
  return files

def _get_list_from_download_link(download_link, filter_text='singleplayer', type=''):
  full_list = _get_list_from_html_file(download_link)
  all_files = [file for file in full_list if filter_text in file and file.endswith(type)]
  return all_files

def _get_full_file_location(file_name, current_location):
  return os.path.join(current_location, file_name)

def _map_download_locations_of_files(files, file_location, json_file_location = None, image_file_location = None, basenames_to_locations_map = {}):
  basenames = [_get_basename(file) for file in files]
  for basename in basenames:
    if json_file_location is not None and basename.endswith('.json'):
      file_path = _get_full_file_location(basename, json_file_location)
    elif image_file_location is not None and basename.endswith('.png'):
      file_path = _get_full_file_location(basename, image_file_location)
    else:
      file_path = _get_full_file_location(basename, file_location)
    if basename not in basenames_to_locations_map:
      basenames_to_locations_map[basename] = file_path
  
  return basenames, basenames_to_locations_map

def _get_download_link_from_files(download_link, files_to_download):
  return [[download_link + '/' + file, file] for file in files_to_download]

def _download_single_file(file_url_to_download, current_location, file_name):
  with urllib3.request('GET', file_url_to_download, preload_content=False) as r, open(_get_full_file_location(file_name, current_location), 'wb') as out_file:       
    shutil.copyfileobj(r, out_file)

def _download_files_direct(download_link, files_to_download, current_location, num_connections=16, start_file=0):
  actual_download_links_and_files = _get_download_link_from_files(download_link, files_to_download)
  with concurrent.futures.ThreadPoolExecutor(max_workers=num_connections) as executor:
    # Download and log every 100 files using a generator
    # First initialize the generator
    current_file = 0
    current_file_log = start_file
    for _ in executor.map(lambda x: _download_single_file(x[0], current_location, x[1]), actual_download_links_and_files):
      current_file += 1
      current_file_log += 1
      if (current_file_log and current_file_log % 100 == 0) or current_file == len(files_to_download):
        print('Downloaded ' + str(current_file_log) + ' files')

def _download_files(download_link, files_to_download, file_location, json_file_location = None, image_file_location = None):
  print('Downloading ' + str(len(files_to_download)) + ' files')
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
    _download_files_direct(download_link, files_to_json_location, json_file_location, start_file=len(files_to_normal_location))
  if len(files_to_image_location):
    _download_files_direct(download_link, files_to_image_location, image_file_location, start_file=len(files_to_normal_location) + len(files_to_json_location))
  pass

def _get_non_downloaded_files_list(remote_files, local_files):
  local_files_set = set(local_files)
  return [file for file in remote_files if file not in local_files_set]

# Separate files into files from previous list (data_list) and new files
def split_files(files, first_files=[]):
  if not len(first_files):
    return files, []
  first_files_set = set(first_files)
  files_set = set(files)
  first_matching_files = sorted(list(files_set & first_files_set))
  second_matching_files = sorted(list(files_set - first_files_set))
  return first_matching_files, second_matching_files

# Shuffle files, but keep files from the previous list (data_list) in the first part and new files in the second part to keep it more stable
def shuffle_files(files, seed, first_files=[]):
  first_files, second_files = split_files(files, first_files)
  random.seed(seed)
  first_files_perm = random.sample(first_files, len(first_files))
  second_files_perm = random.sample(second_files, len(second_files)) if len(second_files) else []
  return first_files_perm + second_files_perm
  
def _process_in_pairs(all_files, type='', limit=None, shuffle_seed=None, additional_order=[]):
  processed_files = []
  if type:
    random_perm_files = shuffle_files(all_files, shuffle_seed if shuffle_seed is not None else 42, additional_order)
    processed_files = random_perm_files[:limit] if limit else random_perm_files
  else:
    # individually and with same seed to keep pairs
    json_files = [file for file in all_files if file.endswith('.json')]
    image_files = [file for file in all_files if file.endswith('.png')]
    random_perm_json_files = shuffle_files(json_files, shuffle_seed if shuffle_seed is not None else 42, additional_order)
    processed_json_files = random_perm_json_files[:limit] if limit else random_perm_json_files
    random_perm_image_files = shuffle_files(image_files, shuffle_seed if shuffle_seed is not None else 42, additional_order)
    processed_image_files = random_perm_image_files[:limit] if limit else random_perm_image_files
    processed_files = processed_json_files + processed_image_files
  return processed_files

def _get_list_from_local_dir(file_location, json_file_location = None, image_file_location = None, filter_text='singleplayer', type='', basenames_to_locations_map={}):
  all_files = []
  if file_location is not None:
    all_files.extend(list([file.path for file in os.scandir(file_location)]))
  if json_file_location is not None and type != 'png':
    json_files = list([file.path for file in os.scandir(json_file_location)])
    json_files = [file for file in json_files if file.endswith('.json')]
    all_files.extend(json_files)
  if image_file_location is not None and type != 'json':
    image_files = list([file.path for file in os.scandir(image_file_location)])
    image_files = [file for file in image_files if file.endswith('.png')]
    all_files.extend(image_files)
    
  all_files = list(filter(lambda x: filter_text in x and x.endswith(type), all_files))
    
  all_files, basenames_to_locations_map = _map_to_basenames(all_files, basenames_to_locations_map)
  
  # Filter None values        
  all_files = [file for file in all_files if file is not None]
  
  print('All local files: ' + str(len(all_files)))
  return all_files, basenames_to_locations_map

def _get_list_from_remote(download_link, file_location, json_file_location = None, image_file_location = None, filter_text='singleplayer', type='', basenames_to_locations_map={}):
  all_files = _get_list_from_download_link(download_link, filter_text, type)
  basenames, basenames_to_locations_map = _map_download_locations_of_files(all_files, file_location, json_file_location, image_file_location, basenames_to_locations_map)
  
  print('All remote files: ' + str(len(all_files)))
  return basenames, basenames_to_locations_map

def _get_files_list(file_location, json_file_location = None, image_file_location = None, filter_text='singleplayer', type='', download_link=None, pre_download=False, from_remote_only=False):
  basenames_to_locations_map={}
  basenames = []
  remote_files = []
  local_files = []
  non_downloaded_files = []
  if download_link is not None:
    remote_files, basenames_to_locations_map = _get_list_from_remote(download_link, file_location, json_file_location, image_file_location, filter_text, type, basenames_to_locations_map)
    basenames.extend(remote_files)
  elif from_remote_only:
    raise ValueError('No download link given')
  if not from_remote_only:
    local_files, basenames_to_locations_map = _get_list_from_local_dir(file_location, json_file_location, image_file_location, filter_text, type, basenames_to_locations_map)
    basenames.extend(local_files)
    
  if len(remote_files):
    non_downloaded_files = _get_non_downloaded_files_list(remote_files, local_files)
    if pre_download and len(non_downloaded_files):
      _download_files(download_link, non_downloaded_files, file_location, json_file_location, image_file_location)
      
  # Remove duplicates
  basenames_dedup = []
  basenames_dedup_set = set()
  for basename in basenames:
    if basename not in basenames_dedup_set:
      basenames_dedup.append(basename)
      basenames_dedup_set.add(basename)
  basenames = basenames_dedup
  
  # Remove unpaired files
  if not type:
    basenames = _remove_unpaired_files(basenames)
    
  print('Relevant files: ' + str(len(basenames)))  
  return basenames, basenames_to_locations_map, non_downloaded_files

def _resolve_env_variable(var, env_name):
  if var == 'env':
    var = os.environ.get(env_name)
  return var

# Get file paths of data to load, using multiple locations and optionally a map.
# If ran for a second time, it will use the previous files and otherwise error.
# The limit will automatically be shuffled (but returned in same order).
# If no shuffle seed is given they will be returned in the original order.
# If not just one type is loaded, the limit will be applied per pair of files.
# The shuffle (if enabled) will also be applied per pair of files.
# If a download link is uses, it will be used instead of the file location and the files will be downloaded to the file location.
# Set the file location to "env" to use the environment variable "FILE_LOCATION" as the file location.
# Set the json file location to "env" to use the environment variable "JSON_FILE_LOCATION" as the json file location.
# Set the image file location to "env" to use the environment variable "IMAGE_FILE_LOCATION" as the image file location.
# Set the download link to "env" to use the environment variable "DOWNLOAD_LINK" as the download link.
# If a countries map is given, the files will automatically be pre-downloaded.
def get_data_to_load(loading_file = './data_list', file_location = os.path.join(os.path.dirname(__file__), '1_data_collection/.data'), json_file_location = None, image_file_location = None, filter_text='singleplayer', type='', limit=0, allow_new_file_creation=True, countries_map=None, allow_missing_in_map=False, passthrough_map=False, shuffle_seed=None, download_link=None, pre_download=False, from_remote_only=False, return_basenames_too=False):
  download_link = _resolve_env_variable(download_link, 'DOWNLOAD_LINK')
  file_location = _resolve_env_variable(file_location, 'FILE_LOCATION')
  json_file_location = _resolve_env_variable(json_file_location, 'JSON_FILE_LOCATION')
  image_file_location = _resolve_env_variable(image_file_location, 'IMAGE_FILE_LOCATION')
  
  basenames, basenames_to_locations_map, downloadable_files = _get_files_list(file_location, json_file_location, image_file_location, filter_text, type, download_link, pre_download or countries_map is not None, from_remote_only)
  
  has_loading_file = False
  files_from_loading_file = []
  try:
    if os.stat(loading_file):
      with open(loading_file, 'r', encoding='utf8') as file:
        files_from_loading_file = file.read().split('\n')
        has_loading_file = True
        if limit and len(files_from_loading_file) < (limit * 2 if not type else limit):
          raise ValueError('Can not set limit higher than the number of files in the loading file, remember that the limit is per pair of files if not just one type is loaded')
  except FileNotFoundError:
    pass
  
  if countries_map and not passthrough_map:
    if download_link is not None and not from_remote_only and not has_loading_file:
      print('Warning: If you add local files, this will not be reproducible, consider setting from_remote_only to True')
    basenames, _ = map_occurrences_to_files(basenames, countries_map, allow_missing=allow_missing_in_map, basenames_to_locations_map=basenames_to_locations_map)
    print('Mapped files: ' + str(len(basenames)))
    
  if limit and len(basenames) < (limit * 2 if not type else limit):
    raise ValueError('Can not set limit higher than the number of files available, remember that the limit is per pair of files if not just one type is loaded')
  
  if limit:
    if download_link is not None and not from_remote_only and not has_loading_file:
      print('Warning: If you add local files, this will not be reproducible, consider setting from_remote_only to True')
    limited_files = _process_in_pairs(basenames, type, limit, shuffle_seed, additional_order=files_from_loading_file)
    basenames = [file for file in basenames if file in limited_files]
    print('Limited files: ' + str(len(basenames)))
  if shuffle_seed is not None:
    if download_link is not None and not from_remote_only and not has_loading_file:
      print('Warning: If you add local files, this will not be reproducible, consider setting from_remote_only to True')
    basenames = _process_in_pairs(basenames, type, None, shuffle_seed, additional_order=files_from_loading_file)
    
  if download_link is not None and not pre_download:
    files_to_download = _get_non_downloaded_files_list(basenames, downloadable_files)
    if len(files_to_download):
      _download_files(download_link, files_to_download, file_location, json_file_location, image_file_location)
    
  full_file_paths = _map_to_locations(basenames, basenames_to_locations_map)

  # if no loading file, use the just discovered files
  files_to_load = basenames
  
  if has_loading_file:
    files_to_load = files_from_loading_file
  elif not allow_new_file_creation:
    raise ValueError('No loading file at location')
      
  if not len(files_to_load):
    raise ValueError('No files to load, did you forget to specify the type? Otherwise unpaired files will be ignored')
  
  if not len(basenames):
    raise ValueError('No files in loading location')
  
  actual_file_locations = []
  
  allowed_missing_files = len(basenames) - limit if limit else 0
  previous_missing_files = 0
  
  for file in files_to_load:
    if file not in basenames:
      previous_missing_files += 1
      if previous_missing_files > allowed_missing_files:
        raise ValueError('Missing file ' + file)
      else:
        continue
    else:
      actual_file_locations.append(full_file_paths[basenames.index(file)])
      
  with open(loading_file, 'w', encoding='utf8') as file:
    file.write('\n'.join(files_to_load))
    
  if return_basenames_too:
    return actual_file_locations, files_to_load
    
  return actual_file_locations

# Update data based on factors
def update_data_to_load(files_to_keep, old_loading_file = './data_list', new_loading_file = './updated_data_list', file_location = os.path.join(os.path.dirname(__file__), '1_data_collection/.data'), json_file_location = None, image_file_location = None, filter_text='singleplayer', type='', limit=0, shuffle_seed=None, download_link=None, from_remote_only=False):
  _, previous_files_to_load = get_data_to_load(old_loading_file, file_location, json_file_location, image_file_location, filter_text, type, limit, allow_new_file_creation=False, shuffle_seed=shuffle_seed, download_link=download_link, from_remote_only=from_remote_only, return_basenames_too=True)
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
      raise ValueError('Missing file ' + file)
      
  if not len(files_to_load):
    raise ValueError('No files to load, did you forget to specify the type? Otherwise unpaired files will be ignored')
      
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
def get_countries_occurrences(loading_file = './countries_map_data_list', file_location = os.path.join(os.path.dirname(__file__), '1_data_collection/.data'), filter_text='multiplayer'):
  multiplayer = get_data_to_load(loading_file=loading_file, file_location=file_location, filter_text=filter_text, type='json')
  # map data
  countries, countries_to_files, files_to_countries, num_games = get_countries_occurrences_from_files(multiplayer)
  return countries, countries_to_files, files_to_countries, num_games
