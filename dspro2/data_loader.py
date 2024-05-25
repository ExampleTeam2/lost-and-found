import os
import json
import concurrent
import math
import random
import re
import subprocess
import shutil
import urllib3
from PIL import Image

DEFAULT_DOWNLOAD_LINK='http://49.12.197.1'

# load .env file
from dotenv import load_dotenv
load_dotenv()

def get_counterpart(file):
  # Get the counterpart of a file (json or png)
  # Get the counterpart
  if file.endswith('.json'):
    counterpart = file.replace('result', 'location').replace('.json', '.png')
  elif file.endswith('.png'):
    counterpart = file.replace('location', 'result').replace('.png', '.json')
  else:
    raise ValueError('Invalid file type', file)
  return counterpart

# Get rid of unpaired files (where only either json or png is present)
def _remove_unpaired_files(files):
  print('Filtering out unpaired files')
  file_dict = {}
  paired_files = []

  # Create a dictionary to track each file and its counterpart's presence
  for file in files:
    counterpart = get_counterpart(file)
    file_dict[file] = counterpart
    file_dict.setdefault(counterpart, None)

  # Collect files where both members of the pair are present
  paired_files = [file for file in files if file_dict[file_dict[file]] is not None]
  
  print('Filtered out ' + str(len(files) - len(paired_files)) + ' unpaired files')
  return paired_files

# Initially converted Bermuda here, but it is just not included in the singleplayer data in the first place (but in the multiplayer data)
country_groups = {}

def get_countries_occurrences_from_files(files, basenames_to_locations_map=None, cached_basenames_to_countries={}):
  # filter out-non json files
  json_files = list(filter(lambda x: x.endswith('.json'), files))
  json_basenames = [_get_basename(file) for file in json_files]
  
  missing_json_files = [file for file, basename in zip(json_files, json_basenames) if basename not in cached_basenames_to_countries]
  
  missing_json_files_full_paths = missing_json_files
  if basenames_to_locations_map is not None:
    missing_json_files_full_paths = _map_to_locations(missing_json_files, basenames_to_locations_map)
    
  # load missing data
  missing_json_data = load_json_files(missing_json_files_full_paths, allow_err=True)
  
  json_data = []
  for file in json_basenames:
    if file in cached_basenames_to_countries:
      country = cached_basenames_to_countries[file]
      # technically 'country' is different from 'country_name' but it doesn't matter here
      json_data.append({'country_name': country, 'country': country })
    else:
      json_data.append(missing_json_data.pop(0))
  
  # get all countries with their number of games
  countries = {}
  countries_to_files = {}
  files_to_countries = {}
  countries_to_basenames = {}
  basenames_to_countries = {}
  for file, game, basename in zip(json_files, json_data, json_basenames):
      if game is None:
        continue
      if 'country_name' not in game:
        if 'country' not in game:
          print('Country not found in game: ' + file)
          continue
        else:
          raise ValueError('Country name not found in game, was not enriched: ' + file)
      country = game['country_name']
      # Convert to actual country
      if country in country_groups:
        country = country_groups[country]
      if country in countries:
          countries[country] += 1
          countries_to_files[country].append(file)
          countries_to_basenames[country].append(basename)
      else:
          countries[country] = 1
          countries_to_files[country] = [file]
          countries_to_basenames[country] = [basename]
      files_to_countries[file] = country
      basenames_to_countries[basename] = country
          
  # sort countries by number of games
  sorted_countries = sorted(countries.items(), key=lambda x: x[1], reverse=True)
  # Update the dict to keep the order
  countries = dict(sorted_countries)
  return countries, countries_to_files, files_to_countries, len(json_data), countries_to_basenames, basenames_to_countries

# Assuming either just json or png is give, also return the others (in a flat list all together)
def _get_files_counterparts(files, all_files):
  files_counterparts = []
  for file in files:
    counterpart = get_counterpart(file)
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

def _map_to_location(file, basenames_to_locations_map):
  return basenames_to_locations_map.get(file, file)

def _map_to_location_or_throw(file, basenames_to_locations_map):
  return basenames_to_locations_map[file]

# From a list of files and a map, restore the original locations
def _map_to_locations(files, basenames_to_locations_map, throw=False):
  return [_map_to_location(file, basenames_to_locations_map) for file in files] if not throw else [_map_to_location_or_throw(file, basenames_to_locations_map) for file in files]

# Takes in a list of files and a occurrence map (from a different_dataset)), create an optimally mapped list of files where the occurrences correspond to the map (or are multiples of them)
def map_occurrences_to_files(files, occurrence_map, countries_map_percentage_threshold, countries_map_slack_factor=None, allow_missing=False, basenames_to_locations_map=None, cached_basenames_to_countries={}):
  # get the occurrences of the files itself
  files_occurrences, _, _, num_files, countries_to_basenames, _ = get_countries_occurrences_from_files(files, basenames_to_locations_map=basenames_to_locations_map, cached_basenames_to_countries=cached_basenames_to_countries)
  original_countries_to_basenames = {country: files for country, files in countries_to_basenames.items()}
  original_occurrences = {country: num for country, num in files_occurrences.items()}
  other_countries_to_basenames = {}
  other_files_occurrences = {}
  if countries_map_percentage_threshold:
    # filter out countries with less than the threshold
    countries_to_basenames = {country: files for country, files in countries_to_basenames.items() if len(files) / num_files >= countries_map_percentage_threshold}
    other_countries_to_basenames = {country: files for country, files in original_countries_to_basenames.items() if country not in countries_to_basenames}
    # and update the files occurrences
    files_occurrences = {country: num for country, num in files_occurrences.items() if country in countries_to_basenames}
    other_files_occurrences = {country: num for country, num in original_occurrences.items() if country in other_countries_to_basenames}
  # get the factors between each of the countries (nan if not in the map)
  all_countries = list(set([*occurrence_map.keys(), *files_occurrences.keys()]))
  factors = [(files_occurrences[country] / occurrence_map[country]) if (country in occurrence_map and country in files_occurrences) else float('nan') for country in all_countries]
  # if any of the factors is nan, raise an exception
  if any([x != x for x in factors]):
    if not allow_missing:
      missing_countries = [country for country, factor in zip(all_countries, factors) if factor != factor]
      # check if any of them are in the occurrence map
      missing_countries = [country for country in missing_countries if country in occurrence_map]
      if len(missing_countries):
        print('Missing countries in the map:', missing_countries)
        raise ValueError('Missing country in one of the maps')
    # filter out the missing countries
    factors = [x for x in factors if x == x]
    print(f'Using {len(factors)} countries (out of {len(all_countries)} options)')
  if allow_missing and len(factors) == 0:
    raise ValueError('No countries in commmon between the maps')
  # Get the lowest factor
  factor = min(factors)
  # Get the number of files to load by country
  new_occurrences = {country: math.ceil(occurrence_map[country] * factor) for country in occurrence_map if country in files_occurrences}
  
  # Optionally add the other countries fitting the slack factor 
  if countries_map_percentage_threshold and countries_map_slack_factor is not None:
    other_countries = list(set([*occurrence_map.keys(), *other_files_occurrences.keys()]))
    other_factors = [(other_files_occurrences[country] / occurrence_map[country]) if (country in occurrence_map and country in other_files_occurrences) else float('nan') for country in other_countries]
    # set other factors to nan if they are below the slack factor
    slacked_factors = [x * countries_map_slack_factor if x == x and ((x / factor) >= countries_map_slack_factor) else float('nan') for x in other_factors]
    # get countries with factors above the slack factor
    other_relevant_countries = [country for country, factor in zip(other_countries, slacked_factors) if factor == factor]
    other_relevant_factors = [factor for factor in slacked_factors if factor == factor]
    print(f'Slack factor included {len(other_relevant_countries)} additional countries (of {len(other_countries)} options)')
    # add to the new occurrences map
    for country, slacked_factor in zip(other_relevant_countries, other_relevant_factors):
      new_occurrences[country] = math.ceil(occurrence_map[country] * slacked_factor)
      
  # Get the files to load
  files_to_load = []
  for country in new_occurrences:
    if allow_missing and (country not in countries_to_basenames and country not in other_countries_to_basenames):
      continue
    country_basenames = countries_to_basenames.get(country, None)
    if country_basenames is None:
      country_basenames = other_countries_to_basenames[country]
    files_to_load.extend(country_basenames[:new_occurrences[country]])
  # Get the pairs of files to load
  files_with_counterparts = _get_files_counterparts(files_to_load, [*files, *countries_to_basenames.values(), *other_countries_to_basenames.values()])
  return files_with_counterparts, len(files_to_load)

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
  all_files = [file for file in full_list if filter_text in file and (file.endswith(type) if type else (file.endswith('.json') or file.endswith('.png')))]
  return all_files

# from geoguessr_location_singleplayer_<id>_<num>.png to <id>_<num>
def _get_file_id(file):
  return "_".join(file.split('_')[-2:]).split('.')[0]

def _get_nested_dir_prefix(file):
  file_id = _get_file_id(file)
  return file_id[0] + '/' + file_id[1] + '/'

def _get_full_file_location(file_name, current_location, nested=False):
  if nested:
    return os.path.join(current_location, _get_nested_dir_prefix(file_name), file_name)
  return os.path.join(current_location, file_name)

def _map_download_locations_of_files(files, file_location, json_file_location = None, image_file_location = None, basenames_to_locations_map = {}, nested=False):
  basenames = [_get_basename(file) for file in files]
  for basename in basenames:
    if json_file_location is not None and basename.endswith('.json'):
      file_path = _get_full_file_location(basename, json_file_location, nested=nested)
    elif image_file_location is not None and basename.endswith('.png'):
      file_path = _get_full_file_location(basename, image_file_location, nested=nested)
    else:
      file_path = _get_full_file_location(basename, file_location, nested=nested)
    # make absolute
    file_path = os.path.abspath(file_path)
    if basename not in basenames_to_locations_map:
      basenames_to_locations_map[basename] = file_path
  
  return basenames, basenames_to_locations_map

def _get_download_link_from_files(download_link, files_to_download):
  return [[download_link + '/' + file, file] for file in files_to_download]

def _download_single_file(file_url_to_download, current_location, file_name, use_files_list=False, nested=False):
  with urllib3.request('GET', file_url_to_download, preload_content=False) as r, open(_get_full_file_location(file_name, current_location, nested=nested), 'wb') as out_file:
    shutil.copyfileobj(r, out_file)
    if use_files_list:
      with open(current_location + '/files_list', 'a') as file:
        file.write(file_name + '\n')

def _download_files_direct(download_link, files_to_download, current_location, num_connections=16, start_file=0, use_files_list=False, nested=False):
  actual_download_links_and_files = _get_download_link_from_files(download_link, files_to_download)
  # make absolute
  current_location = os.path.abspath(current_location)
  with concurrent.futures.ThreadPoolExecutor(max_workers=num_connections) as executor:
    # Download and log every 100 files using a generator
    # First initialize the generator
    current_file = 0
    current_file_log = start_file
    for _ in executor.map(lambda x: _download_single_file(x[0], current_location, x[1], use_files_list=use_files_list, nested=nested), actual_download_links_and_files):
      current_file += 1
      current_file_log += 1
      if (current_file_log and current_file_log % 1000 == 0) or current_file == len(files_to_download):
        print('Downloaded ' + str(current_file_log) + ' files')
        
def _list_dir_contents(file_location):
  # make absolute
  file_location = os.path.abspath(file_location)
  return list([file.path for file in os.scandir(file_location)])

def _filter_dir_contents(files):
  return [file for file in files if 'geoguessr' in file]

def _get_id_dir(file_location, char, file_paths=[], depth=2, get_all=False):
  if (not get_all and os.path.exists(file_location + '/' + char)) or get_all:
    if (depth - 1) > 0:
      file_paths.extend(_get_id_dirs(file_location + '/' + char, depth - 1, get_all))
    else:
      file_paths.append(file_location + '/' + char)
        
  return file_paths
        
        

def _get_id_dirs(file_location, depth=2, get_all=False):
  file_paths = []
  # 0-9
  for i in range(10):
    file_paths = _get_id_dir(file_location, str(i), file_paths, depth, get_all)
        
  # lowercase a-z
  for i in range(26):
    file_paths = _get_id_dir(file_location, chr(97 + i), file_paths, depth, get_all)
        
  # uppercase A-Z
  for i in range(26):
    file_paths = _get_id_dir(file_location, chr(65 + i), file_paths, depth, get_all)
        
  return file_paths

def _create_id_dirs(file_location, depth=2, num_workers=16):
  file_paths = _get_id_dirs(file_location, depth, get_all=True)
  
  # parallelize creating the directories
  with concurrent.futures.ThreadPoolExecutor(num_workers) as executor:
    list(executor.map(lambda x: os.makedirs(x, exist_ok=True), file_paths))
      
def _get_id_dir_contents(file_location, depth=2, num_workers=16):
  file_paths = _get_id_dirs(file_location, depth)
  
  # parallelize getting the files
  with concurrent.futures.ThreadPoolExecutor(num_workers) as executor:
    file_paths = list(executor.map(_list_dir_contents, file_paths))
    file_paths = [file for files in file_paths for file in files]
      
  file_paths = _filter_dir_contents(file_paths)
        
  return file_paths
  
def _download_files(download_link, files_to_download, file_location, json_file_location = None, image_file_location = None, num_connections=16, use_files_list=False, nested=False):
  print('Downloading ' + str(len(files_to_download)) + ' files')
  
  if nested:
    _create_id_dirs(file_location)
    if json_file_location is not None:
      _create_id_dirs(json_file_location)
    if image_file_location is not None:
      _create_id_dirs(image_file_location)
  
  files_to_normal_location = []
  files_to_json_location = []
  files_to_image_location = []
  if json_file_location is not None:
    files_to_json_location = [file for file in files_to_download if file.endswith('.json')]
  else:
    files_to_normal_location.extend([file for file in files_to_download if file.endswith('.json')])
  if image_file_location is not None:
    files_to_image_location = [file for file in files_to_download if file.endswith('.png')]
  else:
    files_to_normal_location.extend([file for file in files_to_download if file.endswith('.png')])
  if len(files_to_normal_location):
    _download_files_direct(download_link, files_to_normal_location, file_location, num_connections=num_connections, use_files_list=use_files_list, nested=nested)
  if len(files_to_json_location):
    _download_files_direct(download_link, files_to_json_location, json_file_location, start_file=len(files_to_normal_location), num_connections=num_connections, use_files_list=use_files_list, nested=nested)
  if len(files_to_image_location):
    _download_files_direct(download_link, files_to_image_location, image_file_location, start_file=len(files_to_normal_location) + len(files_to_json_location), num_connections=num_connections, use_files_list=use_files_list, nested=nested)
  pass

def _get_non_downloaded_files_list(remote_files, local_files):
  local_files_set = set(local_files)
  return [file for file in remote_files if file not in local_files_set]

def _get_downloadable_files_list(wanted_files, downloadable_files):
  downloadable_files_set = set(downloadable_files)
  return [file for file in wanted_files if file in downloadable_files_set]

# Separate files into files from previous list (data_list) and new files
def split_files(files, first_files=[]):
  if not len(first_files):
    return files, []
  first_files_set = set(first_files)
  files_set = set(files)
  first_matching_files = sorted(list(files_set & first_files_set))
  second_matching_files = sorted(list(files_set - first_files_set))
  return first_matching_files, second_matching_files

def shuffle_files_simple(files, seed):
  sorted_files = sorted(files)
  random.seed(seed)
  files_perm = random.sample(sorted_files, len(sorted_files))
  return files_perm

def limit_files_simple(files, limit, seed):
  if seed is None:
    seed = 42
  shuffled_files = shuffle_files_simple(files, seed)
  limited_files = shuffled_files[:limit]
  limited_files_set = set(limited_files)
  return [file for file in files if file in limited_files_set]

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

def process_in_pairs_simple(all_files, type='', limit=None, shuffle_seed=None):
  processed_files = []
  if type:
    random_perm_files = shuffle_files_simple(all_files, shuffle_seed if shuffle_seed is not None else 42)
    processed_files = random_perm_files[:limit] if limit else random_perm_files
  else:
    # individually and with same seed to keep pairs
    json_files = [file for file in all_files if file.endswith('.json')]
    image_files = [file for file in all_files if file.endswith('.png')]
    random_perm_json_files = shuffle_files_simple(json_files, shuffle_seed if shuffle_seed is not None else 42)
    processed_json_files = random_perm_json_files[:limit] if limit else random_perm_json_files
    random_perm_image_files = shuffle_files_simple(image_files, shuffle_seed if shuffle_seed is not None else 42)
    processed_image_files = random_perm_image_files[:limit] if limit else random_perm_image_files
    processed_files = processed_json_files + processed_image_files
  return processed_files

def get_all_files(path, use_files_list=False, nested=False):
  # create the directory if it does not exist
  if not os.path.exists(path):
    os.makedirs(path)
    
  stripped_path = re.sub(r'/$', '', path)
  # make absolute
  if not stripped_path.startswith('/'):
    stripped_path = os.path.abspath(stripped_path)
    
  files_list = ""
    
  if use_files_list:
    if not os.path.exists(stripped_path + '/files_list'):
      # create empty file
      with open(stripped_path + '/files_list', 'w') as file:
        file.write('')
    
    with open(stripped_path + '/files_list', 'r') as file:
      files_list = file.read()
      
  if nested and not use_files_list:
    return _get_id_dir_contents(path)
  
  files = _list_dir_contents(path) if not use_files_list else [stripped_path + '/' + (file if not nested else (_get_nested_dir_prefix(file) + file)) for file in files_list.split('\n') if file]
  return _filter_dir_contents(files)

def get_files(path, use_files_list=False, nested=False):
  files = get_all_files(path, use_files_list=use_files_list, nested=nested)
  return [file for file in files if file.endswith('.json') or file.endswith('.png')]

def get_json_files(path, use_files_list=False, nested=False):
  files = get_all_files(path, use_files_list=use_files_list, nested=nested)
  return [file for file in files if file.endswith('.json')]

def get_image_files(path, use_files_list=False, nested=False):
  files = get_all_files(path, use_files_list=use_files_list, nested=nested)
  return [file for file in files if file.endswith('.png')]

def _get_list_from_local_dir(file_location, json_file_location = None, image_file_location = None, filter_text='singleplayer', type='', basenames_to_locations_map={}, use_files_list=False, nested=False):
  all_files = []
  if file_location is not None:
    all_files.extend(get_files(file_location, use_files_list=use_files_list, nested=nested))
  if json_file_location != file_location and json_file_location is not None and type != 'png':
    all_files.extend(get_json_files(json_file_location, use_files_list=use_files_list, nested=nested))
  if image_file_location != file_location and image_file_location is not None and type != 'json':
    all_files.extend(get_image_files(image_file_location, use_files_list=use_files_list, nested=nested))
    
  all_files = list(filter(lambda x: filter_text in x and x.endswith(type), all_files))
  
  # Filter None values        
  all_files = [file for file in all_files if file is not None]
    
  all_files, basenames_to_locations_map = _map_to_basenames(all_files, basenames_to_locations_map)
  
  print('All local files: ' + str(len(all_files)))
  return all_files, basenames_to_locations_map

def _get_list_from_remote(download_link, file_location, json_file_location = None, image_file_location = None, filter_text='singleplayer', type='', basenames_to_locations_map={}, nested=False):
  all_files = _get_list_from_download_link(download_link, filter_text, type)
  # Filter None values        
  all_files = [file for file in all_files if file is not None]
  
  basenames, basenames_to_locations_map = _map_download_locations_of_files(all_files, file_location, json_file_location, image_file_location, basenames_to_locations_map, nested=nested)
  
  print('All remote files: ' + str(len(all_files)))
  return basenames, basenames_to_locations_map

def _copy_and_unzip_files(path, zip_name, current_dir, tmp_dir_name):
  # Create tmp_dir if it does not exist
  tmp_dir = os.path.join(current_dir, tmp_dir_name)
  if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
  # Check if zip file exists at path, if yes, unzip it into tmp_dir (so all files are in the tmp_dir)
  zip_path = os.path.join(path, zip_name)
  if os.path.exists(zip_path):
    print('Copying and unzipping ' + zip_name)
    shutil.copyfile(zip_path, os.path.join(current_dir, zip_name))
    print('Unzipping ' + zip_name)
    # Unpack into tmp_dir
    shutil.unpack_archive(zip_path, tmp_dir)
    print('Unzipped ' + zip_name)
  files_list_path = os.path.join(path, 'files_list')
  # Copy files_list to tmp_dir if it exists
  if os.path.exists(files_list_path):
    shutil.copyfile(files_list_path, os.path.join(tmp_dir, 'files_list'))
    
def _load_from_zips_to_tmp(file_location, json_file_location = None, image_file_location = None):
  zip_name = 'files.zip'
  current_dir = os.getcwd()
  tmp_dir_name = 'tmp'
  if file_location is not None:
    _copy_and_unzip_files(file_location, zip_name, current_dir, tmp_dir_name)
  if json_file_location != file_location and json_file_location is not None and type != 'png':
    _copy_and_unzip_files(json_file_location, zip_name, current_dir, tmp_dir_name)
  if image_file_location != file_location and image_file_location is not None and type != 'json':
    _copy_and_unzip_files(image_file_location, zip_name, current_dir, tmp_dir_name)
  return os.path.join(current_dir, tmp_dir_name)
    
def _copy_files_list(path, files_list_path):
  # Copy files_list to path if it exists
  shutil.copyfile(files_list_path, os.path.join(path, 'files_list'))
    
def _zip_and_copy_files(path, zip_name, current_dir, tmp_dir_name):
  print('Zipping and copying ' + zip_name)
  # Check if there are any files in tmp_dir
  tmp_dir = os.path.join(current_dir, tmp_dir_name)
  # Zip tmp_dir into current_dir
  print('Zipping ' + zip_name)
  shutil.make_archive(os.path.join(current_dir, zip_name.split('.')[0]), 'zip', tmp_dir)
  # Copy zip to path
  print('Copying ' + zip_name)
  shutil.copyfile(os.path.join(current_dir, zip_name), os.path.join(path, zip_name))
  print('Copied ' + zip_name)
  
def _save_to_zips_from_tmp(file_location, json_file_location = None, image_file_location = None):
  zip_name = 'files.zip'
  current_dir = os.getcwd()
  tmp_dir_name = 'tmp'
  files_list_path = os.path.join(current_dir, 'files_list')
  if os.path.exists(files_list_path):
    if file_location is not None:
      _copy_files_list(file_location, files_list_path)
    if json_file_location != file_location and json_file_location is not None and type != 'png':
      _copy_files_list(json_file_location, files_list_path)
    if image_file_location != file_location and image_file_location is not None and type != 'json':
      _copy_files_list(image_file_location, files_list_path)
    # Remove files_list
    os.remove(files_list_path)
      
  if file_location is not None:
    _zip_and_copy_files(file_location, zip_name, current_dir, tmp_dir_name)
  if json_file_location != file_location and json_file_location is not None and type != 'png':
    _zip_and_copy_files(json_file_location, zip_name, current_dir, tmp_dir_name)
  if image_file_location != file_location and image_file_location is not None and type != 'json':
    _zip_and_copy_files(image_file_location, zip_name, current_dir, tmp_dir_name)

def _get_files_list(file_location, json_file_location = None, image_file_location = None, filter_text='singleplayer', type='', download_link=None, pre_download=False, from_remote_only=False, dedupe_and_remove_unpaired=True, skip_checks=False, num_download_connections=16, use_files_list=False, nested=False, tmp_dir_and_zip=False):
  basenames_to_locations_map={}
  basenames = []
  remote_files = []
  local_files = []
  non_downloaded_files = []
  if download_link is not None:
    remote_files, basenames_to_locations_map = _get_list_from_remote(download_link, file_location, json_file_location, image_file_location, filter_text, type, basenames_to_locations_map, nested=nested)
    basenames.extend(remote_files)
  elif from_remote_only:
    raise ValueError('No download link given')
  local_files, basenames_to_locations_map = _get_list_from_local_dir(file_location, json_file_location, image_file_location, filter_text, type, basenames_to_locations_map, use_files_list=use_files_list, nested=nested)
  if not from_remote_only:  
    basenames.extend(local_files)
    
  pre_downloaded_new_files = False
  if len(remote_files):
    non_downloaded_files = _get_non_downloaded_files_list(remote_files, local_files)
    if pre_download and len(non_downloaded_files):
      pre_downloaded_new_files = True
      _download_files(download_link, non_downloaded_files, file_location, json_file_location, image_file_location, num_connections=num_download_connections, use_files_list=use_files_list, nested=nested)
      
  if dedupe_and_remove_unpaired and not skip_checks:
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
  return basenames, basenames_to_locations_map, non_downloaded_files, pre_downloaded_new_files

def resolve_env_variable(var, env_name, do_not_enforce_but_allow_env=None, alt_env=None):
  if var == 'env' or do_not_enforce_but_allow_env == True and var is not None:
    if do_not_enforce_but_allow_env == False:
      raise ValueError('Prefer providing a default file location and setting <name>_allow_env=True')
    new_var = os.environ.get(env_name)
    if new_var is None and alt_env is not None:
      new_var = os.environ.get(alt_env)
    if do_not_enforce_but_allow_env is None and new_var is None:
      raise ValueError('Environment variable ' + env_name + ' not set')
    elif new_var is not None:
      return str(new_var)
  return var

# Get file paths of data to load, using multiple locations and optionally a map.
# If ran for a second time, it will use the previous files and otherwise error.
# The limit will automatically be shuffled (but returned in same order).
# If no shuffle seed is given they will be returned in the original order.
# If not just one type is loaded, the limit will be applied per pair of files.
# The shuffle (if enabled) will also be applied per pair of files.
# If a download link is used, it will be used instead of the file location and the files will be downloaded to the file location.
# Set the allow_file_location_env=True to use the environment variable "FILE_LOCATION" as the file location, otherwise file_location will be used.
# Set the json file location to "env" to use the environment variable "JSON_FILE_LOCATION" as the json file location, otherwise json_file_location will be used.
# Set the image file location to "env" to use the environment variable "IMAGE_FILE_LOCATION" as the image file location, otherwise image_file_location will be used.
# Set the download link to "default" to use the default, set allow_download_link_env=True to use the environment variable "DOWNLOAD_LINK" as the download link.
# Set the environment variable "SKIP_REMOTE" to "true" to skip the remote files and only use the local files. (even if from_remote_only is set), only use this if you are sure the current files are already downloaded.
# Set the environment variable "SKIP_CHECKS" to "true" to skip all of the checks and just use the files from the data-list. Only use this if you are sure the files are already downloaded and structured correctly.
# Set allow_new_file_creation=False to only allow loading from the loading file, otherwise an error will be raised. This will improve loading performance.
# If a countries map is given, the files will automatically be pre-downloaded.
# The countries_map_percentage_threshold is the minimum percentage of games (of the total) a country should have to be included in the map, it only works if allow_missing_in_map is set to True.
# If countries_map_slack_factor is set (only works if countries_map_percentage_threshold is set), it will allow countries to be included in the map if they are within the slack factor of the percentage threshold. This can also be set to 1 to include countries that can be mapped but do not match countries_map_percentage_threshold.
def get_data_to_load(loading_file = './data_list', file_location = os.path.join(os.path.dirname(__file__), '1_data_collection/.data'), json_file_location = None, image_file_location = None, filter_text='singleplayer', type='', limit=0, allow_new_file_creation=True, countries_map=None, countries_map_percentage_threshold=0, countries_map_slack_factor=None, allow_missing_in_map=False, passthrough_map=False, shuffle_seed=None, download_link=None, pre_download=False, from_remote_only=False, allow_file_location_env=False, allow_json_file_location_env=False, allow_image_file_location_env=False, allow_download_link_env=False, num_download_connections=16, allow_num_download_connections_env=True, countries_map_cached_basenames_to_countries={}, return_basenames_too=False):
  if download_link == 'default':
    download_link = DEFAULT_DOWNLOAD_LINK
  download_link = resolve_env_variable(download_link, 'DOWNLOAD_LINK', allow_download_link_env)
  file_location = resolve_env_variable(file_location, 'FILE_LOCATION', allow_file_location_env)
  json_file_location = resolve_env_variable(json_file_location, 'JSON_FILE_LOCATION', allow_json_file_location_env)
  image_file_location = resolve_env_variable(image_file_location, 'IMAGE_FILE_LOCATION', allow_image_file_location_env)
  skip_remote = resolve_env_variable(str(False), 'SKIP_REMOTE', True)
  skip_remote = skip_remote is not None and skip_remote and skip_remote.lower() != 'false' and skip_remote.lower() != '0'
  skip_checks = resolve_env_variable(str(False), 'SKIP_CHECKS', True)
  skip_checks = skip_checks is not None and skip_checks and skip_checks.lower() != 'false' and skip_checks.lower() != '0'
  num_download_connections = int(resolve_env_variable(str(num_download_connections), 'NUM_DOWNLOAD_CONNECTIONS', allow_num_download_connections_env))
  use_files_list = resolve_env_variable(str(False), 'USE_FILES_LIST', True)
  use_files_list = use_files_list is not None and use_files_list and use_files_list.lower() != 'false' and use_files_list.lower() != '0'
  nested = resolve_env_variable(str(True), 'NESTED', True)
  nested = not (nested is not None and nested and nested.lower() != 'true' and nested.lower() != '1')
  tmp_dir_and_zip = resolve_env_variable(str(False), 'TMP_DIR_AND_ZIP', True)
  tmp_dir_and_zip = tmp_dir_and_zip is not None and tmp_dir_and_zip and tmp_dir_and_zip.lower() != 'false' and tmp_dir_and_zip.lower() != '0'
  if skip_checks:
    print('Warning: Skipping all checks')
    skip_remote = True
  elif skip_remote and download_link:
    print('Warning: Skipping remote files check')
  if skip_remote:
    from_remote_only = False
    download_link = None

  pre_download = pre_download or countries_map is not None
  
  original_file_location = file_location
  original_json_file_location = json_file_location
  original_image_file_location = image_file_location
  tmp_dir = None
  if tmp_dir_and_zip:
    tmp_dir = _load_from_zips_to_tmp(file_location, json_file_location, image_file_location)
    file_location = tmp_dir
    json_file_location = tmp_dir
    image_file_location = tmp_dir
  
  basenames, basenames_to_locations_map, downloadable_files, pre_downloaded_new_files = _get_files_list(file_location, json_file_location, image_file_location, filter_text, type, download_link, pre_download, from_remote_only, allow_new_file_creation, skip_checks, num_download_connections=num_download_connections, use_files_list=use_files_list, nested=nested)
  downloaded_new_files = pre_downloaded_new_files
  
  has_loading_file = False
  files_from_loading_file = []
  try:
    if os.stat(loading_file):
      with open(loading_file, 'r', encoding='utf8') as file:
        files_from_loading_file = file.read().split('\n')
        # filter None values and empty strings
        files_from_loading_file = [file for file in files_from_loading_file if file is not None and file]
        if skip_checks:
          # skip all checks and logic
          basenames = files_from_loading_file
        has_loading_file = True
        if limit and len(files_from_loading_file) < (limit * 2 if not type else limit):
          raise ValueError('Can not set limit higher than the number of files in the loading file, remember that the limit is per pair of files if not just one type is loaded')
  except FileNotFoundError:
    if skip_checks:
      raise ValueError('No loading file at location, but checks are skipped')
    pass
  
  if countries_map and not passthrough_map:
    if skip_checks:
      raise ValueError('Countries map given, but checks are skipped')
    if download_link is not None and not from_remote_only and not has_loading_file:
      print('Warning: If you add local files, this will not be reproducible, consider setting from_remote_only to True')
    mapped_files, _ = map_occurrences_to_files(basenames, countries_map, countries_map_percentage_threshold, countries_map_slack_factor, allow_missing=allow_missing_in_map, basenames_to_locations_map=basenames_to_locations_map, cached_basenames_to_countries=countries_map_cached_basenames_to_countries)
    mapped_files_set = set(mapped_files)
    basenames = [file for file in basenames if file in mapped_files_set]
    print('Mapped files: ' + str(len(basenames)))
    
  if limit and len(basenames) < (limit * 2 if not type else limit):
    raise ValueError('Can not set limit higher than the number of files available, remember that the limit is per pair of files if not just one type is loaded')
  
  if limit:
    if download_link is not None and not from_remote_only and not has_loading_file:
      print('Warning: If you add local files, this will not be reproducible, consider setting from_remote_only to True')
    limited_files = _process_in_pairs(basenames, type, limit, shuffle_seed, additional_order=files_from_loading_file) if not skip_checks else process_in_pairs_simple(basenames, type, limit, shuffle_seed)
    limited_files_set = set(limited_files)
    basenames = [file for file in basenames if file in limited_files_set]
    print('Limited files: ' + str(len(basenames)))
  if shuffle_seed is not None:
    if download_link is not None and not from_remote_only and not has_loading_file:
      print('Warning: If you add local files, this will not be reproducible, consider setting from_remote_only to True')
    basenames = _process_in_pairs(basenames, type, None, shuffle_seed, additional_order=files_from_loading_file) if not skip_checks else process_in_pairs_simple(basenames, type, None, shuffle_seed)
    
  if download_link is not None and not pre_download and len(downloadable_files):
    files_to_download = _get_downloadable_files_list(basenames, downloadable_files)
    if len(files_to_download) and has_loading_file:
      downloadable_files_from_loading_file = _get_downloadable_files_list(files_from_loading_file, downloadable_files)
      files_to_download = _get_downloadable_files_list(basenames, downloadable_files_from_loading_file)
    if len(files_to_download):
      downloaded_new_files = True
      _download_files(download_link, files_to_download, file_location, json_file_location, image_file_location, num_connections=num_download_connections, use_files_list=use_files_list, nested=nested)
      
  if tmp_dir_and_zip:
    file_location = original_file_location
    json_file_location = original_json_file_location
    image_file_location = original_image_file_location
    if downloaded_new_files:
      _save_to_zips_from_tmp(file_location, json_file_location, image_file_location)
      
      
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
  
  allowed_missing_files = len(files_to_load) - limit if limit else 0
  previous_missing_files = 0
  
  basenames_set = set(basenames)
  
  if skip_checks:
    actual_file_locations = _map_to_locations(files_to_load, basenames_to_locations_map, throw=True)
    if return_basenames_too:
      return actual_file_locations, files_to_load
    return actual_file_locations
  
  for file in files_to_load:
    if file not in basenames_set:
      previous_missing_files += 1
      if previous_missing_files > allowed_missing_files:
        raise ValueError('Missing file ' + file)
      else:
        continue
    else:
      actual_file_locations.append(_map_to_location_or_throw(file, basenames_to_locations_map))
      
  with open(loading_file, 'w', encoding='utf8') as file:
    file.write('\n'.join(files_to_load))
    
  if return_basenames_too:
    return actual_file_locations, files_to_load
    
  return actual_file_locations

# Update data based on factors
def update_data_to_load(files_to_keep, old_loading_file = './data_list', new_loading_file = './updated_data_list', file_location = os.path.join(os.path.dirname(__file__), '1_data_collection/.data'), json_file_location = None, image_file_location = None, filter_text='singleplayer', type='', limit=0, shuffle_seed=None, download_link=None, from_remote_only=False, allow_file_location_env=False, allow_json_file_location_env=False, allow_image_file_location_env=False, allow_download_link_env=False, num_download_connections=16):
  _, previous_files_to_load = get_data_to_load(old_loading_file, file_location, json_file_location, image_file_location, filter_text, type, limit, allow_new_file_creation=False, shuffle_seed=shuffle_seed, download_link=download_link, from_remote_only=from_remote_only, allow_file_location_env=allow_file_location_env, allow_json_file_location_env=allow_json_file_location_env, allow_image_file_location_env=allow_image_file_location_env, allow_download_link_env=allow_download_link_env, num_download_connections=num_download_connections, return_basenames_too=True)
  files_to_load = []
  base_files_to_keep = set([os.path.basename(file) for file in files_to_keep])
  for previous_file_to_load in previous_files_to_load:
    if previous_file_to_load in base_files_to_keep:
      files_to_load.append(previous_file_to_load)
     
  base_files = set(files_to_load)
      
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

def split_json_and_image_files(files):
  json_files = [file for file in files if file.endswith('.json')]
  image_files = [file for file in files if file.endswith('.png')]
  return json_files, image_files

# load a single json file
def load_json_file(file, allow_err=False):
  with open(file, 'r', encoding='utf8') as f:
    try:
      return json.load(f)
    except json.JSONDecodeError as error:
      if allow_err: 
        print(f"JSONDecodeError in {os.path.basename(file)}")
        print(error)
        return None
      else:
        raise error

# load mutliple json files parallelized
def load_json_files(files, num_workers=16, allow_err=False):
  with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    results = list(executor.map(lambda f: load_json_file(f, allow_err), files))
  return results  

# load a single .png file as a converted image
def load_image_file(file):
  # channels, height, width is the pytorch convention
  with Image.open(file) as img:
    return img.convert('RGB')

# load mutliple .png files parallelized
def load_image_files(files, num_workers=16):
  with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    results = list(executor.map(load_image_file, files))
  return results

# load a single .png file, needs to be closed manually or used in a with statement
def load_image_file_raw(file):
  # channels, height, width is the pytorch convention
  return Image.open(file)

# load mutliple .png files parallelized
def load_image_files_raw(files, num_workers=16):
  with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    results = list(executor.map(load_image_file_raw, files))
  return results

# get countries occurrences from games
def get_countries_occurrences(loading_file = './countries_map_data_list', file_location = os.path.join(os.path.dirname(__file__), '1_data_collection/.data'), filter_text='multiplayer', download_link=None, from_remote_only=False, allow_file_location_env=False, allow_image_file_location_env=False, allow_json_file_location_env=False, allow_download_link_env=False, num_download_connections=16):
  files = get_data_to_load(loading_file=loading_file, file_location=file_location, filter_text=filter_text, type='json', download_link=download_link, from_remote_only=from_remote_only, allow_file_location_env=allow_file_location_env, allow_image_file_location_env=allow_image_file_location_env, allow_json_file_location_env=allow_json_file_location_env, allow_download_link_env=allow_download_link_env, num_download_connections=num_download_connections)
  # map data
  countries, countries_to_files, files_to_countries, num_games, countries_to_basenames, basenames_to_countries = get_countries_occurrences_from_files(files)
  return countries, countries_to_files, files_to_countries, num_games, countries_to_basenames, basenames_to_countries
