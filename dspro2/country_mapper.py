import math
import os
from data_loader import get_basename, get_data_to_load, get_files_counterparts, map_to_locations, load_json_files


# Initially converted Bermuda here, but it is just not included in the singleplayer data in the first place (but in the multiplayer data)
country_groups = {}


def get_countries_occurrences_from_files(files, basenames_to_locations_map=None, cached_basenames_to_countries={}):
    # filter out-non json files
    json_files = list(filter(lambda x: x.endswith(".json"), files))
    json_basenames = [get_basename(file) for file in json_files]

    missing_json_files = [file for file, basename in zip(json_files, json_basenames) if basename not in cached_basenames_to_countries]

    missing_json_files_full_paths = missing_json_files
    if basenames_to_locations_map is not None:
        missing_json_files_full_paths = map_to_locations(missing_json_files, basenames_to_locations_map)

    # load missing data
    missing_json_data = load_json_files(missing_json_files_full_paths, allow_err=True)

    json_data = []
    for file in json_basenames:
        if file in cached_basenames_to_countries:
            country = cached_basenames_to_countries[file]
            # technically 'country' is different from 'country_name' but it doesn't matter here
            json_data.append({"country_name": country, "country": country})
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
        if "country_name" not in game:
            if "country" not in game:
                print("Country not found in game: " + file)
                continue
            else:
                raise ValueError("Country name not found in game, was not enriched: " + file)
        country = game["country_name"]
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


# Takes in a list of files and a occurrence map (from a different_dataset)), create an optimally mapped list of files where the occurrences correspond to the map (or are multiples of them)
def map_countries_occurrences_to_files(files, occurrence_map, countries_map_percentage_threshold, countries_map_slack_factor=None, allow_missing=False, basenames_to_locations_map=None, cached_basenames_to_countries={}):
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
    factors = [(files_occurrences[country] / occurrence_map[country]) if (country in occurrence_map and country in files_occurrences) else float("nan") for country in all_countries]
    # if any of the factors is nan, raise an exception
    if any([x != x for x in factors]):
        if not allow_missing:
            missing_countries = [country for country, factor in zip(all_countries, factors) if factor != factor]
            # check if any of them are in the occurrence map
            missing_countries = [country for country in missing_countries if country in occurrence_map]
            if len(missing_countries):
                print("Missing countries in the map:", missing_countries)
                raise ValueError("Missing country in one of the maps")
        # filter out the missing countries
        factors = [x for x in factors if x == x]
        print(f"Using {len(factors)} countries (out of {len(all_countries)} options)")
    if allow_missing and len(factors) == 0:
        raise ValueError("No countries in commmon between the maps")
    # Get the lowest factor
    factor = min(factors)
    # Get the number of files to load by country
    new_occurrences = {country: math.ceil(occurrence_map[country] * factor) for country in occurrence_map if country in files_occurrences}

    # Optionally add the other countries fitting the slack factor
    if countries_map_percentage_threshold and countries_map_slack_factor is not None:
        other_countries = list(set([*occurrence_map.keys(), *other_files_occurrences.keys()]))
        other_factors = [(other_files_occurrences[country] / occurrence_map[country]) if (country in occurrence_map and country in other_files_occurrences) else float("nan") for country in other_countries]
        # set other factors to nan if they are below the slack factor
        slacked_factors = [x * countries_map_slack_factor if x == x and ((x / factor) >= countries_map_slack_factor) else float("nan") for x in other_factors]
        # get countries with factors above the slack factor
        other_relevant_countries = [country for country, factor in zip(other_countries, slacked_factors) if factor == factor]
        other_relevant_factors = [factor for factor in slacked_factors if factor == factor]
        print(f"Slack factor included {len(other_relevant_countries)} additional countries (of {len(other_countries)} options)")
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
        files_to_load.extend(country_basenames[: new_occurrences[country]])
    # Get the pairs of files to load
    files_with_counterparts = get_files_counterparts(files_to_load, [*files, *countries_to_basenames.values(), *other_countries_to_basenames.values()])
    return files_with_counterparts, len(files_to_load)


# Get countries occurrences from games
# Works like the get_data_to_load of the data_loader, but returns the countries and their occurrences.
def get_countries_occurrences(loading_file="./countries_map_data_list", file_location=os.path.join(os.path.dirname(__file__), "1_data_collection/.data"), filter_text="multiplayer", download_link=None, from_remote_only=False, allow_file_location_env=False, allow_image_file_location_env=False, allow_json_file_location_env=False, allow_download_link_env=False, num_download_connections=16):
    files = get_data_to_load(loading_file=loading_file, file_location=file_location, filter_text=filter_text, type="json", download_link=download_link, from_remote_only=from_remote_only, allow_file_location_env=allow_file_location_env, allow_image_file_location_env=allow_image_file_location_env, allow_json_file_location_env=allow_json_file_location_env, allow_download_link_env=allow_download_link_env, num_download_connections=num_download_connections)
    # map data
    countries, countries_to_files, files_to_countries, num_games, countries_to_basenames, basenames_to_countries = get_countries_occurrences_from_files(files)
    return countries, countries_to_files, files_to_countries, num_games, countries_to_basenames, basenames_to_countries


# Get a mapper to be used with the data_loader get_data_to_load function.
# The countries_map_percentage_threshold is the minimum percentage of games (of the total) the class (country) should have to be included in the map, it only works if allow_missing_in_map is set to True.
# If countries_map_slack_factor is set (only works if countries_map_percentage_threshold is set), it will allow countries to be included in the map if they are within the slack factor of the percentage threshold. This can also be set to 1 to include countries that can be mapped but do not match countries_map_percentage_threshold.
def get_mapper(countries_map=None, countries_map_percentage_threshold=0, countries_map_slack_factor=None, allow_missing_in_map=False, countries_map_cached_basenames_to_countries={}):
    lambda basenames, basenames_to_locations_map: map_countries_occurrences_to_files(basenames, countries_map, countries_map_percentage_threshold, countries_map_slack_factor, allow_missing=allow_missing_in_map, basenames_to_locations_map=basenames_to_locations_map, cached_basenames_to_countries=countries_map_cached_basenames_to_countries)
