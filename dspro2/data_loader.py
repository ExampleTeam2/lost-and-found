import os
import json
import concurrent
import math
import random
import re
import shutil
import urllib3
import hashlib
from PIL import Image

DEFAULT_DOWNLOAD_LINK = "http://49.12.197.1"

# If this is in the file name, it is a geoguessr file, should be set
FILE_NAME_PART = "geoguessr"

# If not specified otherwise, expect this text to be in the file name, could also be empty
DEFAULT_SINGLEPLAYER_FILTER_TEXT = "singleplayer"

# If this is in the file name, it is a json file, could also be empty
JSON_FILE_NAME_PART = "result"

# If this is in the file name, it is a png file, could also be empty
IMAGE_FILE_NAME_PART = "location"

# load .env file
from dotenv import load_dotenv

load_dotenv()


def get_counterpart(file):
    # Get the counterpart of a file (json or png)
    # Get the counterpart
    if file.endswith(".json"):
        counterpart = file.replace(JSON_FILE_NAME_PART, IMAGE_FILE_NAME_PART).replace(".json", ".png")
    elif file.endswith(".png"):
        counterpart = file.replace(IMAGE_FILE_NAME_PART, JSON_FILE_NAME_PART).replace(".png", ".json")
    else:
        raise ValueError("Invalid file type", file)
    return counterpart


def hash_filenames(file_names):
    # Create a SHA-256 hash object
    hash_object = hashlib.sha256()

    # Concatenate all file names into one long string and update the hash object
    for name in file_names:
        # Ensure encoding to bytes, as hashlib requires bytes input
        hash_object.update(get_basename(name).encode("utf-8"))

    # Get the hexadecimal representation of the hash
    hash_digest = hash_object.hexdigest()
    return hash_digest


def _get_tmp_dir():
    tmp_dir_and_zip = resolve_env_variable(str(False), "TMP_DIR_AND_ZIP", True)
    tmp_dir_and_zip = tmp_dir_and_zip is not None and tmp_dir_and_zip and tmp_dir_and_zip.lower() != "false" and tmp_dir_and_zip.lower() != "0"

    current_dir = os.getcwd()

    tmp_dir = None
    if tmp_dir_and_zip:
        tmp_dir_name = "tmp"
        tmp_dir = os.path.join(current_dir, tmp_dir_name)

    return tmp_dir, tmp_dir_and_zip, current_dir


# Get the file path for caching processed data
def get_cached_file_path(file_names, config, name="data", suffix=".pth"):
    file_name = hash_filenames(file_names) + suffix
    config_keys = reversed(sorted(list(config.keys())))
    first = True
    for key in config_keys:
        file_name = key + "=" + str(config[key]) + ("&" if first else "") + file_name
        first = False

    file_name = name + "_" + str(len(file_names) // 2) + "_" + file_name

    tmp_dir, _, current_dir = _get_tmp_dir()
    dir_to_check = tmp_dir if tmp_dir is not None else current_dir

    return os.path.join(dir_to_check, file_name)


# Get the file path of cached processed data if it exists
def potentially_get_cached_file_path(file_names, config, name="data", suffix=".pth", cache_load_callback=None):
    file_path = get_cached_file_path(file_names, config, name=name, suffix=suffix)
    if cache_load_callback is not None:
        file_name = get_basename(file_path)
        print("Checking remote cache for " + file_name)
        cache_load_callback(file_name)
    if os.path.exists(file_path):
        return file_path
    return None


# Get rid of unpaired files (where only either json or png is present)
def _remove_unpaired_files(files):
    print("Filtering out unpaired files")
    file_dict = {}
    paired_files = []

    # Create a dictionary to track each file and its counterpart's presence
    for file in files:
        counterpart = get_counterpart(file)
        file_dict[file] = counterpart
        file_dict.setdefault(counterpart, None)

    # Collect files where both members of the pair are present
    paired_files = [file for file in files if file_dict[file_dict[file]] is not None]

    print("Filtered out " + str(len(files) - len(paired_files)) + " unpaired files")
    return paired_files


# Assuming either just json or png is give, also return the others (in a flat list all together)
def get_files_counterparts(files, all_files):
    files_counterparts = []
    for file in files:
        counterpart = get_counterpart(file)
        if counterpart in all_files:
            files_counterparts.append(counterpart)
            files_counterparts.append(file)
        else:
            files_counterparts.append(file)
    return files_counterparts


# Get the file name from a full path
def get_basename(file):
    return os.path.basename(file)


# Replace one start of the file with another and create a map from the original files to the new ones, then return the new files and the map
def _map_to_basenames(files, basenames_to_locations_map={}):
    basenames = [get_basename(file) for file in files]
    for file, basename in zip(files, basenames):
        if basename not in basenames_to_locations_map:
            basenames_to_locations_map[basename] = file
    return basenames, basenames_to_locations_map


def _map_to_location(file, basenames_to_locations_map):
    return basenames_to_locations_map.get(file, file)


def _map_to_location_or_throw(file, basenames_to_locations_map):
    return basenames_to_locations_map[file]


# From a list of files and a map, restore the original locations
def map_to_locations(files, basenames_to_locations_map, throw=False):
    return [_map_to_location(file, basenames_to_locations_map) for file in files] if not throw else [_map_to_location_or_throw(file, basenames_to_locations_map) for file in files]


def _get_list_from_html_file(download_link):
    # get list of files from nginx or apache html file
    http = urllib3.PoolManager()
    print("Getting files list from remote")
    try:
        response = http.request("GET", download_link)
        html = response.data.decode("utf-8")
    except Exception as e:
        if download_link == DEFAULT_DOWNLOAD_LINK:
            print("Either your connection to the internet is restricted or disrupted, or our remote server is no longer available, please use your own server.")
            raise ConnectionError("Either your connection to the internet is restricted or disrupted, or our remote server is no longer available, please use your own server or train locally.")
        raise e
    print("Got files list from remote")
    files = []
    if not html:
        raise ValueError("No response from remote server")
    for line in html.split("<a"):
        if "href=" in line:
            if not 'href="' in line and '"' in line[len("href=") :]:
                raise ValueError("Invalid line in remote response: " + line)
            file = line.split('href="')[-1].split('"')[0]
            if not file:
                raise ValueError("Invalid link in remote response: " + line.split('href="')[-1])
            file_name = file.split("/")[-1] if "/" in file else file
            if file_name:
                files.append(file_name)
    files = [file for file in files if file is not None]
    print("Parsed files list from remote")
    return files


def _get_list_from_download_link(download_link, filter_text=DEFAULT_SINGLEPLAYER_FILTER_TEXT, type=""):
    full_list = _get_list_from_html_file(download_link)
    all_files = [file for file in full_list if filter_text in file and (file.endswith(type) if type else (file.endswith(".json") or file.endswith(".png")))]
    return all_files


# from geoguessr_location_singleplayer_<id>_<num>.png (or similar) to <id>_<num>
def _get_file_id(file):
    return "_".join(file.split("_")[-2:]).split(".")[0]


def _get_nested_dir_prefix(file):
    file_id = _get_file_id(file)
    return file_id[0] + "/" + file_id[1] + "/"


def _get_full_file_location(file_name, current_location, nested=False):
    if nested:
        return os.path.join(current_location, _get_nested_dir_prefix(file_name), file_name)
    return os.path.join(current_location, file_name)


def _map_download_locations_of_files(files, file_location, json_file_location=None, image_file_location=None, basenames_to_locations_map={}, nested=False):
    basenames = [get_basename(file) for file in files]
    for basename in basenames:
        if json_file_location is not None and basename.endswith(".json"):
            file_path = _get_full_file_location(basename, json_file_location, nested=nested)
        elif image_file_location is not None and basename.endswith(".png"):
            file_path = _get_full_file_location(basename, image_file_location, nested=nested)
        else:
            file_path = _get_full_file_location(basename, file_location, nested=nested)
        # make absolute
        file_path = os.path.abspath(file_path)
        if basename not in basenames_to_locations_map:
            basenames_to_locations_map[basename] = file_path

    return basenames, basenames_to_locations_map


def _get_download_link_from_files(download_link, files_to_download):
    return [[download_link + "/" + file, file] for file in files_to_download]


def _download_single_file(file_url_to_download, current_location, file_name, use_files_list=False, nested=False):
    with urllib3.request("GET", file_url_to_download, preload_content=False) as r, open(_get_full_file_location(file_name, current_location, nested=nested), "wb") as out_file:
        shutil.copyfileobj(r, out_file)
        if use_files_list:
            with open(current_location + "/files_list", "a") as file:
                file.write(file_name + "\n")


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
                print("Downloaded " + str(current_file_log) + " files")


def _list_dir_contents(file_location):
    # make absolute
    file_location = os.path.abspath(file_location)
    return list([file.path for file in os.scandir(file_location)])


def _filter_dir_contents(files):
    return [file for file in files if FILE_NAME_PART in file]


def _get_id_dir(file_location, char, file_paths=[], depth=2, get_all=False):
    if (not get_all and os.path.exists(file_location + "/" + char)) or get_all:
        if (depth - 1) > 0:
            file_paths.extend(_get_id_dirs(file_location + "/" + char, depth - 1, get_all))
        else:
            file_paths.append(file_location + "/" + char)

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


def _download_files(download_link, files_to_download, file_location, json_file_location=None, image_file_location=None, num_connections=16, use_files_list=False, nested=False):
    print("Downloading " + str(len(files_to_download)) + " files")

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
        files_to_json_location = [file for file in files_to_download if file.endswith(".json")]
    else:
        files_to_normal_location.extend([file for file in files_to_download if file.endswith(".json")])
    if image_file_location is not None:
        files_to_image_location = [file for file in files_to_download if file.endswith(".png")]
    else:
        files_to_normal_location.extend([file for file in files_to_download if file.endswith(".png")])
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


def _process_in_pairs(all_files, type="", limit=None, shuffle_seed=None, additional_order=[]):
    processed_files = []
    if type:
        random_perm_files = shuffle_files(all_files, shuffle_seed if shuffle_seed is not None else 42, additional_order)
        processed_files = random_perm_files[:limit] if limit else random_perm_files
    else:
        # individually and with same seed to keep pairs
        json_files = [file for file in all_files if file.endswith(".json")]
        image_files = [file for file in all_files if file.endswith(".png")]
        random_perm_json_files = shuffle_files(json_files, shuffle_seed if shuffle_seed is not None else 42, additional_order)
        processed_json_files = random_perm_json_files[:limit] if limit else random_perm_json_files
        random_perm_image_files = shuffle_files(image_files, shuffle_seed if shuffle_seed is not None else 42, additional_order)
        processed_image_files = random_perm_image_files[:limit] if limit else random_perm_image_files
        processed_files = processed_json_files + processed_image_files
    return processed_files


def process_in_pairs_simple(all_files, type="", limit=None, shuffle_seed=None):
    processed_files = []
    if type:
        random_perm_files = shuffle_files_simple(all_files, shuffle_seed if shuffle_seed is not None else 42)
        processed_files = random_perm_files[:limit] if limit else random_perm_files
    else:
        # individually and with same seed to keep pairs
        json_files = [file for file in all_files if file.endswith(".json")]
        image_files = [file for file in all_files if file.endswith(".png")]
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

    stripped_path = re.sub(r"/$", "", path)
    # make absolute
    if not stripped_path.startswith("/"):
        stripped_path = os.path.abspath(stripped_path)

    files_list = ""
    found_files_list = False

    if use_files_list:
        if os.path.exists(stripped_path + "/files_list"):
            found_files_list = True
            with open(stripped_path + "/files_list", "r") as file:
                files_list = file.read()

    final_files = []
    if nested and (not use_files_list or not found_files_list):
        final_files = _get_id_dir_contents(path)
    else:
        files = _list_dir_contents(path) if not use_files_list or not found_files_list else [stripped_path + "/" + (file if not nested else (_get_nested_dir_prefix(file) + file)) for file in files_list.split("\n") if file]
        final_files = _filter_dir_contents(files)

    if use_files_list and not found_files_list:
        # create file with all files
        basenames = [get_basename(file) for file in final_files]
        with open(stripped_path + "/files_list", "w") as file:
            file.write("\n".join(basenames) + "\n")

    return final_files


def get_files(path, use_files_list=False, nested=False):
    files = get_all_files(path, use_files_list=use_files_list, nested=nested)
    return [file for file in files if file.endswith(".json") or file.endswith(".png")]


def get_json_files(path, use_files_list=False, nested=False):
    files = get_all_files(path, use_files_list=use_files_list, nested=nested)
    return [file for file in files if file.endswith(".json")]


def get_image_files(path, use_files_list=False, nested=False):
    files = get_all_files(path, use_files_list=use_files_list, nested=nested)
    return [file for file in files if file.endswith(".png")]


def _get_list_from_local_dir(file_location, json_file_location=None, image_file_location=None, filter_text=DEFAULT_SINGLEPLAYER_FILTER_TEXT, type="", basenames_to_locations_map={}, use_files_list=False, nested=False):
    all_files = []
    if file_location is not None:
        all_files.extend(get_files(file_location, use_files_list=use_files_list, nested=nested))
    if json_file_location != file_location and json_file_location is not None and type != "png":
        all_files.extend(get_json_files(json_file_location, use_files_list=use_files_list, nested=nested))
    if image_file_location != file_location and image_file_location is not None and type != "json":
        all_files.extend(get_image_files(image_file_location, use_files_list=use_files_list, nested=nested))

    all_files = list(filter(lambda x: filter_text in x and x.endswith(type), all_files))

    # Filter None values
    all_files = [file for file in all_files if file is not None]

    all_files, basenames_to_locations_map = _map_to_basenames(all_files, basenames_to_locations_map)

    print("All local files: " + str(len(all_files)))
    return all_files, basenames_to_locations_map


def _get_list_from_remote(download_link, file_location, json_file_location=None, image_file_location=None, filter_text=DEFAULT_SINGLEPLAYER_FILTER_TEXT, type="", basenames_to_locations_map={}, nested=False):
    all_files = _get_list_from_download_link(download_link, filter_text, type)
    # Filter None values
    all_files = [file for file in all_files if file is not None]

    basenames, basenames_to_locations_map = _map_download_locations_of_files(all_files, file_location, json_file_location, image_file_location, basenames_to_locations_map, nested=nested)

    print("All remote files: " + str(len(all_files)))
    return basenames, basenames_to_locations_map


def _copy_and_unzip_files(path, zip_name, current_dir, tmp_dir="./tmp", always_load_zip=False, only_load_zip=False, only_load_pth=None):
    # Create tmp_dir if it does not exist
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    if not os.path.exists(path):
        os.makedirs(path)
    skip_zip = False
    loaded_zip = False
    # Load only the specified pth file if it is specified
    if only_load_pth is not None:
        file = only_load_pth
        if os.path.exists(os.path.join(path, file)):
            # Skip it if it is already in the tmp_dir
            if os.path.exists(os.path.join(tmp_dir, file)):
                print("Skipping copying " + file + " because it is already in the tmp_dir")
                return loaded_zip
            # Delete all other .pth files
            for delete_file in os.listdir(tmp_dir):
                if delete_file.endswith(".pth"):
                    print("Deleting " + delete_file)
                    os.remove(os.path.join(tmp_dir, delete_file))
                    print("Deleted " + delete_file)
            # Copy file to tmp_dir
            print("Copying " + file)
            shutil.copyfile(os.path.join(path, file), os.path.join(tmp_dir, file))
            print("Copied " + file)
        return loaded_zip
    if not only_load_zip:
        # copy all .wandb files to tmp_dir if they exist
        for file in os.listdir(path):
            if file.endswith(".pth"):
                skip_zip = True
                # Defer loading the zip file to later
            elif file.endswith(".wandb"):
                # Copy file to tmp_dir
                print("Copying " + file)
                shutil.copyfile(os.path.join(path, file), os.path.join(tmp_dir, file))
                print("Copied " + file)
    if not skip_zip or always_load_zip:
        loaded_zip = True
        # Check if zip file exists at path, if yes, unzip it into tmp_dir (so all files are in the tmp_dir)
        zip_path = os.path.join(path, zip_name)
        if os.path.exists(zip_path) or os.path.exists(os.path.join(current_dir, zip_name)):
            if not os.path.exists(os.path.join(current_dir, zip_name)):
                print("Copying and unzipping " + zip_name)
                shutil.copyfile(zip_path, os.path.join(current_dir, zip_name))
            print("Unzipping " + zip_name)
            # Unpack into tmp_dir
            shutil.unpack_archive(zip_path, tmp_dir)
            print("Unzipped " + zip_name)
    else:
        print("Skipped copying and unzipping " + zip_name)

    files_list_path = os.path.join(path, "files_list")
    # Copy files_list to tmp_dir if it exists
    if os.path.exists(files_list_path):
        shutil.copyfile(files_list_path, os.path.join(tmp_dir, "files_list"))

    return loaded_zip


def _load_from_zips_to_tmp(file_location, json_file_location=None, image_file_location=None, current_dir="./", tmp_dir="./tmp", always_load_zip=False, only_load_zip=False, only_load_pth=None):
    zip_name = "files.zip"
    loaded_zip = False
    if file_location is not None:
        loaded_zip = _copy_and_unzip_files(file_location, zip_name, current_dir, tmp_dir, always_load_zip=always_load_zip, only_load_zip=only_load_zip, only_load_pth=only_load_pth) or loaded_zip
    if json_file_location != file_location and json_file_location is not None and type != "png":
        loaded_zip = _copy_and_unzip_files(json_file_location, zip_name, current_dir, tmp_dir, always_load_zip=always_load_zip, only_load_zip=only_load_zip, only_load_pth=only_load_pth) or loaded_zip
    if image_file_location != file_location and image_file_location is not None and type != "json":
        loaded_zip = _copy_and_unzip_files(image_file_location, zip_name, current_dir, tmp_dir, always_load_zip=always_load_zip, only_load_zip=only_load_zip, only_load_pth=only_load_pth) or loaded_zip

    return loaded_zip


def _copy_other_files(path, files_list_path, additional_files_paths, move=False):
    if files_list_path is not None:
        # Copy files_list to path if it exists
        shutil.copyfile(files_list_path, os.path.join(path, "files_list"))
    # Copy .pth and .wandb files to path if they exist
    for file in additional_files_paths:
        if file.endswith(".pth"):
            # Copy file to path
            pth_basename = get_basename(file)
            # If it has a long file name (100+) (probably unique), skip it if it is already in the path
            if len(pth_basename) > 100 and os.path.exists(os.path.join(path, pth_basename)):
                print("Skipping copying " + pth_basename + " because it is already in the path")
                continue
            print("Copying " + pth_basename)
            if not move:
                shutil.copyfile(file, os.path.join(path, pth_basename))
            else:
                print("Moving " + pth_basename + " instead of copying")
                shutil.move(file, os.path.join(path, pth_basename))
            print("Copied " + pth_basename)
        elif file.endswith(".wandb"):
            # Copy file to path
            wandb_basename = get_basename(file)
            print("Copying " + wandb_basename)
            if move:
                print("Copying " + wandb_basename + " anyway because it is very small, and it could be used later")
            shutil.copyfile(file, os.path.join(path, wandb_basename))
            print("Copied " + wandb_basename)


def _zip_and_copy_files(path, zip_name, current_dir, tmp_dir="./tmp", move=False):
    print("Zipping and copying " + zip_name)
    # Check if there are any files in tmp_dir
    # Zip tmp_dir into current_dir
    print("Zipping " + zip_name)
    shutil.make_archive(os.path.join(current_dir, zip_name.split(".")[0]), "zip", tmp_dir)
    # Copy zip to path
    print("Copying " + zip_name)
    if not move:
        shutil.copyfile(os.path.join(current_dir, zip_name), os.path.join(path, zip_name))
    else:
        print("Moving " + zip_name + " instead of copying")
        shutil.move(os.path.join(current_dir, zip_name), os.path.join(path, zip_name))
    print("Copied " + zip_name)


def _save_to_zips_from_tmp(file_location, json_file_location=None, image_file_location=None, current_dir="./", tmp_dir="./tmp", only_additional=False, skip_additional=True, move=False):
    zip_name = "files.zip"
    current_dir = os.getcwd()
    files_list_path = os.path.join(tmp_dir, "files_list") if not only_additional else None
    if files_list_path is not None:
        if not os.path.exists(files_list_path):
            files_list_path = None
    additional_files_paths = [os.path.join(tmp_dir, file) for file in os.listdir(tmp_dir) if file.endswith(".pth") or file.endswith(".wandb")] if not skip_additional else []
    if files_list_path is not None or len(additional_files_paths):
        if file_location is not None:
            _copy_other_files(file_location, files_list_path, additional_files_paths, move=move)
        if json_file_location != file_location and json_file_location is not None and type != "png":
            _copy_other_files(json_file_location, files_list_path, additional_files_paths, move=move)
        if image_file_location != file_location and image_file_location is not None and type != "json":
            _copy_other_files(image_file_location, files_list_path, additional_files_paths, move=move)
        if files_list_path is not None:
            # Remove files_list
            os.remove(files_list_path)
        # Remove .pth and .wandb files
        for file in additional_files_paths:
            os.remove(file)

    # If there are no files left in tmp_dir, remove it
    skip_zip = False
    if only_additional or (not len(os.listdir(tmp_dir))):
        skip_zip = True

    if not skip_zip:
        if file_location is not None:
            _zip_and_copy_files(file_location, zip_name, current_dir, tmp_dir, move=move)
        if json_file_location != file_location and json_file_location is not None and type != "png":
            _zip_and_copy_files(json_file_location, zip_name, current_dir, tmp_dir, move=move)
        if image_file_location != file_location and image_file_location is not None and type != "json":
            _zip_and_copy_files(image_file_location, zip_name, current_dir, tmp_dir, move=move)
    else:
        print("Skipped zipping and copying " + zip_name)


def _get_files_list(file_location, json_file_location=None, image_file_location=None, filter_text=DEFAULT_SINGLEPLAYER_FILTER_TEXT, type="", download_link=None, pre_download=False, from_remote_only=False, dedupe_and_remove_unpaired=True, skip_checks=False, num_download_connections=16, use_files_list=False, nested=False, tmp_dir_and_zip=False):
    basenames_to_locations_map = {}
    basenames = []
    remote_files = []
    local_files = []
    non_downloaded_files = []
    if download_link is not None:
        remote_files, basenames_to_locations_map = _get_list_from_remote(download_link, file_location, json_file_location, image_file_location, filter_text, type, basenames_to_locations_map, nested=nested)
        basenames.extend(remote_files)
    elif from_remote_only:
        raise ValueError("No download link given")
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

    print("Relevant files: " + str(len(basenames)))
    return basenames, basenames_to_locations_map, non_downloaded_files, pre_downloaded_new_files


def resolve_env_variable(var, env_name, do_not_enforce_but_allow_env=None, alt_env=None, set_none=False):
    if var == "env" or do_not_enforce_but_allow_env == True and var is not None:
        if do_not_enforce_but_allow_env == False:
            raise ValueError("Prefer providing a default file location and setting <name>_allow_env=True")
        new_var = os.environ.get(env_name)
        if new_var is None and alt_env is not None:
            new_var = os.environ.get(alt_env)
        if do_not_enforce_but_allow_env is None and new_var is None:
            raise ValueError("Environment variable " + env_name + " not set")
        if set_none and new_var is not None and (new_var.lower() == "None" or len(new_var) == 0):
            new_var = None
        elif new_var is not None:
            return str(new_var)
    return var


def _get_file_locations(file_location, json_file_location=None, image_file_location=None, allow_file_location_env=False, allow_json_file_location_env=False, allow_image_file_location_env=False):
    file_location = resolve_env_variable(file_location, "FILE_LOCATION", allow_file_location_env)
    json_file_location = resolve_env_variable(json_file_location, "JSON_FILE_LOCATION", allow_json_file_location_env)
    image_file_location = resolve_env_variable(image_file_location, "IMAGE_FILE_LOCATION", allow_image_file_location_env)
    use_files_list = resolve_env_variable(str(False), "USE_FILES_LIST", True)
    use_files_list = use_files_list is not None and use_files_list and use_files_list.lower() != "false" and use_files_list.lower() != "0"
    nested = resolve_env_variable(str(False), "NESTED", True)
    nested = not (nested is not None and nested and nested.lower() != "true" and nested.lower() != "1")
    tmp_dir, tmp_dir_and_zip, current_dir = _get_tmp_dir()
    return file_location, json_file_location, image_file_location, tmp_dir, current_dir, use_files_list, nested, tmp_dir_and_zip


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
# If a map_occurrences_to_files function is given (like the one returned by get_mapper in country_mapper), the files will be mapped and the files will automatically be pre-downloaded.
# If `NESTED=true` is set, the files will be loaded from and saved into nested directories, this is useful for large datasets.
# If `USE_FILES_LIST=true` is set, the names of the files will be loaded from and saved into a files_list file in the directory, this is useful for large datasets.
# If `TMP_DIR_AND_ZIP=true` is set, the files will be loaded from a zip file into the tmp directory and saved to a zip file into the tmp directory. This is useful for large datasets and slow file systems like Google Colab.
# In that case if in the file_location `.pth` files are found, they will be copied to the tmp directory and copied back after the files are loaded. This is used for caching.
# If this is true and `USE_FILES_LIST=true` is set (and not mapping or pre-downloading), the copying of the `.zip` file will be skipped if `.pth` files are found.
# If return_load_and_additional_save_callback is set to True, a callback function will be returned that can be used to load the zip file or a specified .pth file later if required, as well as a callback function that can be used to save the `.pth` files.
def get_data_to_load(loading_file="./data_list", file_location=os.path.join(os.path.dirname(__file__), "1_data_collection/.data"), json_file_location=None, image_file_location=None, filter_text=DEFAULT_SINGLEPLAYER_FILTER_TEXT, type="", limit=0, allow_new_file_creation=True, map_occurrences_to_files=None, passthrough_map=False, shuffle_seed=None, download_link=None, pre_download=False, from_remote_only=False, allow_file_location_env=False, allow_json_file_location_env=False, allow_image_file_location_env=False, allow_download_link_env=False, num_download_connections=16, allow_num_download_connections_env=True, return_basenames_too=False, return_load_and_additional_save_callback=False):
    if download_link == "default":
        download_link = DEFAULT_DOWNLOAD_LINK
    download_link = resolve_env_variable(download_link, "DOWNLOAD_LINK", allow_download_link_env, None, True)
    if download_link == DEFAULT_DOWNLOAD_LINK:
        print("Warning: Downloading from our server will soon no longer be supported, please use local data (DOWNLOAD_LINK=None and `SKIP_REMOTE=true` in .env), the dataset is accessible at https://www.kaggle.com/datasets/killusions/street-location-images/ (put unzipped files into 1_data_collection/.data and run yarn data:import on a unix based system, then the import.ipynb notebook), provide a different download link (DOWNLOAD_LINK in .env) or the scaping script can be used to collect your own data.")
    skip_remote = resolve_env_variable(str(False), "SKIP_REMOTE", True)
    skip_remote = skip_remote is not None and skip_remote and skip_remote.lower() != "false" and skip_remote.lower() != "0"
    skip_checks = resolve_env_variable(str(False), "SKIP_CHECKS", True)
    skip_checks = skip_checks is not None and skip_checks and skip_checks.lower() != "false" and skip_checks.lower() != "0"
    num_download_connections = int(resolve_env_variable(str(num_download_connections), "NUM_DOWNLOAD_CONNECTIONS", allow_num_download_connections_env))
    if skip_checks:
        print("Warning: Skipping all checks")
        skip_remote = True
    elif skip_remote and download_link:
        print("Warning: Skipping remote files check")
    if skip_remote:
        from_remote_only = False
        download_link = None

    pre_download = pre_download or map_occurrences_to_files is not None
    always_load_zip = pre_download

    file_location, json_file_location, image_file_location, tmp_dir, current_dir, use_files_list, nested, tmp_dir_and_zip = _get_file_locations(file_location, json_file_location, image_file_location, allow_file_location_env, allow_json_file_location_env, allow_image_file_location_env)

    original_file_location = file_location
    original_json_file_location = json_file_location
    original_image_file_location = image_file_location

    def get_load_callback(pth_file_name=None):
        _load_from_zips_to_tmp(file_location, json_file_location, image_file_location, current_dir, tmp_dir, always_load_zip=always_load_zip, only_load_zip=True, only_load_pth=pth_file_name)

    def get_only_load_pth_callback(pth_file_name=None):
        if pth_file_name is None:
            return
        get_load_callback(pth_file_name=pth_file_name)

    if not use_files_list:
        always_load_zip = True
    loaded_zip = False
    load_callback = None
    if tmp_dir_and_zip:
        loaded_zip = _load_from_zips_to_tmp(file_location, json_file_location, image_file_location, current_dir, tmp_dir, always_load_zip=always_load_zip)
        file_location = tmp_dir
        json_file_location = tmp_dir
        image_file_location = tmp_dir
        if return_load_and_additional_save_callback:
            load_callback = get_load_callback if not loaded_zip else get_only_load_pth_callback

    basenames, basenames_to_locations_map, downloadable_files, pre_downloaded_new_files = _get_files_list(file_location, json_file_location, image_file_location, filter_text, type, download_link, pre_download, from_remote_only, allow_new_file_creation, skip_checks, num_download_connections=num_download_connections, use_files_list=use_files_list, nested=nested)
    downloaded_new_files = pre_downloaded_new_files

    has_loading_file = False
    files_from_loading_file = []
    try:
        if os.stat(loading_file):
            with open(loading_file, "r", encoding="utf8") as file:
                files_from_loading_file = file.read().split("\n")
                # filter None values and empty strings
                files_from_loading_file = [file for file in files_from_loading_file if file is not None and file]
                if skip_checks:
                    # skip all checks and logic
                    basenames = files_from_loading_file
                has_loading_file = True
                if limit and len(files_from_loading_file) < (limit * 2 if not type else limit):
                    raise ValueError("Can not set limit higher than the number of files in the loading file, remember that the limit is per pair of files if not just one type is loaded")
    except FileNotFoundError:
        if skip_checks:
            raise ValueError("No loading file at location, but checks are skipped")
        pass

    if map_occurrences_to_files and not passthrough_map:
        if skip_checks:
            raise ValueError("Map function given, but checks are skipped")
        if download_link is not None and not from_remote_only and not has_loading_file:
            print("Warning: If you add local files, this will not be reproducible, consider setting from_remote_only to True")
        mapped_files, _ = map_occurrences_to_files(basenames, basenames_to_locations_map=basenames_to_locations_map)
        mapped_files_set = set(mapped_files)
        basenames = [file for file in basenames if file in mapped_files_set]
        print("Mapped files: " + str(len(basenames)))

    if limit and len(basenames) < (limit * 2 if not type else limit):
        raise ValueError("Can not set limit higher than the number of files available, remember that the limit is per pair of files if not just one type is loaded")

    if limit and len(basenames) > (limit * 2 if not type else limit):
        if download_link is not None and not from_remote_only and not has_loading_file:
            print("Warning: If you add local files, this will not be reproducible, consider setting from_remote_only to True")
        limited_files = _process_in_pairs(basenames, type, limit, shuffle_seed, additional_order=files_from_loading_file) if not skip_checks else process_in_pairs_simple(basenames, type, limit, shuffle_seed)
        limited_files_set = set(limited_files)
        basenames = [file for file in basenames if file in limited_files_set]
        print("Limited files: " + str(len(basenames)))
    if shuffle_seed is not None:
        if download_link is not None and not from_remote_only and not has_loading_file:
            print("Warning: If you add local files, this will not be reproducible, consider setting from_remote_only to True")
        basenames = _process_in_pairs(basenames, type, None, shuffle_seed, additional_order=files_from_loading_file) if not skip_checks else process_in_pairs_simple(basenames, type, None, shuffle_seed)

    if download_link is not None and not pre_download and len(downloadable_files):
        files_to_download = _get_downloadable_files_list(basenames, downloadable_files)
        if len(files_to_download) and has_loading_file:
            downloadable_files_from_loading_file = _get_downloadable_files_list(files_from_loading_file, downloadable_files)
            files_to_download = _get_downloadable_files_list(basenames, downloadable_files_from_loading_file)
        if len(files_to_download):
            downloaded_new_files = True
            # If there is a zip file, download the previous files
            if load_callback:
                load_callback()
                load_callback = get_only_load_pth_callback
            _download_files(download_link, files_to_download, file_location, json_file_location, image_file_location, num_connections=num_download_connections, use_files_list=use_files_list, nested=nested)

    additional_save_callback = None

    if tmp_dir_and_zip:
        file_location = original_file_location
        json_file_location = original_json_file_location
        image_file_location = original_image_file_location
        if return_load_and_additional_save_callback:
            additional_save_callback = lambda move=False: _save_to_zips_from_tmp(file_location, json_file_location, image_file_location, current_dir, tmp_dir, only_additional=True, skip_additional=False, move=move)
        if downloaded_new_files:
            _save_to_zips_from_tmp(file_location, json_file_location, image_file_location, current_dir, tmp_dir, only_additional=False, skip_additional=True)
        # delete zip file if it was loaded
        zip_path = os.path.join(current_dir, "files.zip")
        if os.path.exists(zip_path):
            os.remove(zip_path)

    # if no loading file, use the just discovered files
    files_to_load = basenames

    if has_loading_file:
        files_to_load = files_from_loading_file
    elif not allow_new_file_creation:
        print("No loading file at location, disable allow_new_file_creation to create a new one, afterwards commit this file for reproducibility")
        raise ValueError("No loading file at location, disable allow_new_file_creation to create a new one, afterwards commit this file for reproducibility")

    if not len(files_to_load):
        raise ValueError("No files to load, did you forget to specify the type? Otherwise unpaired files will be ignored")

    if not len(basenames):
        raise ValueError("No files in loading location")

    actual_file_locations = []

    allowed_missing_files = len(files_to_load) - limit if limit else 0
    previous_missing_files = 0

    basenames_set = set(basenames)

    if skip_checks:
        actual_file_locations = map_to_locations(files_to_load, basenames_to_locations_map, throw=True)
        if return_basenames_too:
            if return_load_and_additional_save_callback:
                return actual_file_locations, files_to_load, load_callback, additional_save_callback
            return actual_file_locations, files_to_load
        if return_load_and_additional_save_callback:
            return actual_file_locations, load_callback, additional_save_callback
        return actual_file_locations

    for file in files_to_load:
        if file not in basenames_set:
            previous_missing_files += 1
            if previous_missing_files > allowed_missing_files:
                print("Missing file " + file + ", to use just your own local files make sure allow_new_file_creation is enabled and delete " + loading_file + ", afterwards commit this file for reproducibility")
                raise ValueError("Missing file " + file + ", to use just your own local files make sure allow_new_file_creation is enabled and delete " + loading_file + ", afterwards commit this file for reproducibility")
            else:
                continue
        else:
            actual_file_locations.append(_map_to_location_or_throw(file, basenames_to_locations_map))

    with open(loading_file, "w", encoding="utf8") as file:
        file.write("\n".join(files_to_load))

    if return_basenames_too:
        if return_load_and_additional_save_callback:
            return actual_file_locations, files_to_load, load_callback, additional_save_callback
        return actual_file_locations, files_to_load

    if return_load_and_additional_save_callback:
        return actual_file_locations, load_callback, additional_save_callback

    return actual_file_locations


# Update data based on factors
def update_data_to_load(files_to_keep, old_loading_file="./data_list", new_loading_file="./updated_data_list", file_location=os.path.join(os.path.dirname(__file__), "1_data_collection/.data"), json_file_location=None, image_file_location=None, filter_text=DEFAULT_SINGLEPLAYER_FILTER_TEXT, type="", limit=0, shuffle_seed=None, download_link=None, from_remote_only=False, allow_file_location_env=False, allow_json_file_location_env=False, allow_image_file_location_env=False, allow_download_link_env=False, num_download_connections=16):
    _, previous_files_to_load = get_data_to_load(old_loading_file, file_location, json_file_location, image_file_location, filter_text, type, limit, allow_new_file_creation=False, shuffle_seed=shuffle_seed, download_link=download_link, from_remote_only=from_remote_only, allow_file_location_env=allow_file_location_env, allow_json_file_location_env=allow_json_file_location_env, allow_image_file_location_env=allow_image_file_location_env, allow_download_link_env=allow_download_link_env, num_download_connections=num_download_connections, return_basenames_too=True)
    files_to_load = []
    base_files_to_keep = set([get_basename(file) for file in files_to_keep])
    for previous_file_to_load in previous_files_to_load:
        if previous_file_to_load in base_files_to_keep:
            files_to_load.append(previous_file_to_load)

    base_files = set(files_to_load)

    try:
        if os.stat(new_loading_file):
            with open(new_loading_file, "r", encoding="utf8") as file:
                files_to_load = file.read().split("\n")
    except FileNotFoundError:
        pass

    for file in files_to_load:
        if file not in base_files:
            raise ValueError("Missing file " + file)

    if not len(files_to_load):
        raise ValueError("No files to load, did you forget to specify the type? Otherwise unpaired files will be ignored")

    with open(new_loading_file, "w", encoding="utf8") as file:
        file.write("\n".join(files_to_load))


def split_json_and_image_files(files):
    json_files = [file for file in files if file.endswith(".json")]
    image_files = [file for file in files if file.endswith(".png")]
    return json_files, image_files


# load a single json file
def load_json_file(file, allow_err=False):
    with open(file, "r", encoding="utf8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as error:
            if allow_err:
                print(f"JSONDecodeError in {get_basename(file)}")
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
        return img.convert("RGB")


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
