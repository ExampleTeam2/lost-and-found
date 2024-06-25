import numpy as np

default_radius = 6371  # Earth's radius in kilometers

def coordinates_to_cartesian(lat, lon, R=default_radius):
  # Convert degrees to radians
  lon_rad = np.radians(lon)
  lat_rad = np.radians(lat)

  # Cartesian coordinates using numpy
  x = R * np.cos(lat_rad) * np.cos(lon_rad)
  y = R * np.cos(lat_rad) * np.sin(lon_rad)
  z = R * np.sin(lat_rad)
  return np.stack([x, y, z], axis=-1)  # Ensure the output is a numpy array with the correct shape

def cartesian_to_coordinates(x, y, z):
  # Compute radial distances
  radial = np.sqrt(x**2 + y**2 + z**2)

  # Normalize if the radius is not 1
  x_norm = x / radial
  y_norm = y / radial
  z_norm = z / radial

  # Latitude from z coordinate
  lat_rad = np.arcsin(z_norm)
  lat = np.degrees(lat_rad)

  # Longitude from x and y coordinates
  lon_rad = np.arctan2(y_norm, x_norm)
  lon = np.degrees(lon_rad)

  return (lat, lon)
