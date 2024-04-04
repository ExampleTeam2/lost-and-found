import os

def get_data_to_load(loading_file = './data_list', file_location = os.path.join(os.path.dirname(__file__), 'data'), filterText="singleplayer", limit=0):
  all_files = list([file.path for file in os.scandir(file_location)])
  if filterText:
    all_files = list(filter(lambda x: filterText in x, all_files))
  if limit:
    all_files = all_files[:limit]
  base_files = list([os.path.basename(file) for file in all_files])
  files_to_load = base_files
  
  try:
    if os.stat(loading_file):
      with open(loading_file, 'r') as file:
        files_to_load = file.read().split('\n')
  except FileNotFoundError:
    pass
      
  if not len(files_to_load):
    raise ValueError("No files to load")
  
  if not len(all_files):
    raise ValueError("No files in loading location")
  
  actual_file_locations = []
  
  for file in files_to_load:
    if file not in base_files:
      raise ValueError("Missing file " + file)
    else:
      actual_file_locations.append(all_files[base_files.index(file)])
      
  with open(loading_file, 'w') as file:
    file.write("\n".join(files_to_load))
    
  return actual_file_locations
