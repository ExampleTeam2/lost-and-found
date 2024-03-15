import os
import re
import json
import pycountry
import reverse_geocoder as rg

class CountryEnricher:
    """Enriches JSON files with country information based on coordinates."""
    def __init__(self, input_dir, output_dir):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.json_files = {}
        self.coordinates = [] # Batching for faster results, 5 x times faster now
        self.file_map = {} # Batching for faster results, 5 x times faster now
        self.pattern = r'singleplayer_(.+?)\.json'
        self.process()
    
    def load_and_prepare_files(self):
        json_paths = [f for f in os.listdir(self.input_dir) if f.endswith('.json') and 'singleplayer' in f]
        for file in json_paths:
            match = re.search(self.pattern, file)
            if match:
                image_id = match.group(1)
                with open(os.path.join(self.input_dir, file), 'r') as f:
                    json_data = json.load(f)
                self.json_files[image_id] = json_data
                if 'coordinates' in json_data:
                    coord = tuple(json_data['coordinates'])
                    self.coordinates.append(coord)
                    self.file_map[coord] = image_id
    
    def enrich_with_country_info(self):
        results = rg.search(self.coordinates)
        for result, coord in zip(results, self.coordinates):
            image_id = self.file_map[coord]
            country_code = result['cc']
            country = pycountry.countries.get(alpha_2=country_code)
            country_name = country.name if country else 'Unknown'
            self.json_files[image_id]['country_name'] = country_name
            self.json_files[image_id]['country_code'] = country_code
    
    def save_enriched_files(self):
        os.makedirs(self.output_dir, exist_ok=True)
        for image_id, data in self.json_files.items():
            file_name = "geoguessr_result_singleplayer_" + image_id + ".json"
            file_path = os.path.join(self.output_dir, file_name)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
    
    def process(self):
        self.load_and_prepare_files()
        self.enrich_with_country_info()
        self.save_enriched_files()
