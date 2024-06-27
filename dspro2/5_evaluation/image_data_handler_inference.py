import json
import requests


class InferenceImageDataHandler:
    def __init__(self, country_to_index_path="./country_to_index.json", region_to_index_path="./region_to_index.json", region_index_to_middle_point_path="./region_index_to_middle_point.json", region_index_to_country_index_path="./region_index_to_country_index.json"):
        self.num_regions = 0
        self.num_countries = 0

        self.country_to_index = None
        self.region_to_index = None
        self.region_index_to_middle_point = None
        self.region_index_to_country_index = None
        if country_to_index_path is not None:
            # Load the country_to_index mapping
            json_response = requests.get(country_to_index_path)
            json_response.raise_for_status()  # Check if the download was successful

            # Load the file into a variable
            self.country_to_index = json.loads(json_response.text)

            print(f"Loaded {len(self.country_to_index)} countries.")

            self.num_countries = len(self.country_to_index)

        if region_to_index_path is not None:
            # Load the region_to_index mapping
            json_response = requests.get(region_to_index_path)
            json_response.raise_for_status()

            # Load the file into a variable
            self.region_to_index = json.loads(json_response.text)

            print(f"Loaded {len(self.region_to_index)} regions.")

            self.num_regions = len(self.region_to_index)

        if region_index_to_middle_point_path is not None:
            # Load the region_index_to_middle_point mapping
            json_response = requests.get(region_index_to_middle_point_path)
            json_response.raise_for_status()

            print(f"Loaded {len(json.loads(json_response.text))} region middle points.")

            # Load the file into a variable
            self.region_index_to_middle_point = json.loads(json_response.text)
            # Convert the keys to integers
            self.region_index_to_middle_point = {int(k): v for k, v in self.region_index_to_middle_point.items()}

        if region_index_to_country_index_path is not None:
            # Load the region_index_to_country_index mapping
            json_response = requests.get(region_index_to_country_index_path)
            json_response.raise_for_status()

            print(f"Loaded {len(json.loads(json_response.text))} region to country index mappings.")

            # Load the file into a variable
            self.region_index_to_country_index = json.loads(json_response.text)
            # Convert the keys to integers
            self.region_index_to_country_index = {int(k): v for k, v in self.region_index_to_country_index.items()}
