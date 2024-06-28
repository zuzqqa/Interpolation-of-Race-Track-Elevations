import requests
import numpy as np

API_KEY = '...'


def minutes_to_degrees(minutes):
    """Converts minutes to degrees."""
    return minutes / 60


def extract_coordinates(file_path):
    """Extracts coordinates from a file."""
    coordinates = []

    with open(file_path, 'r') as file:
        line = file.readline()
        while line != '[data]\n':
            line = file.readline()

        for line in file:
            values = line.split()
            lat_minutes = float(values[2])
            long_minutes = float(values[3])

            latitude = minutes_to_degrees(lat_minutes)
            longitude = minutes_to_degrees(long_minutes)

            if lat_minutes < 0:
                latitude *= -1
            if long_minutes < 0:
                longitude *= -1

            coordinates.append((latitude, longitude))

    return coordinates


def get_elevation(coordinates):
    """Gets elevation data from Google Maps API."""
    elevations = []
    base_url = "https://maps.googleapis.com/maps/api/elevation/json"
    chunk_size = 512  # Google Maps API allows up to 512 locations per request

    for i in range(0, len(coordinates), chunk_size):
        chunk = coordinates[i:i + chunk_size]
        locations = "|".join([f"{lat},{lon}" for lat, lon in chunk])
        params = {
            "locations": locations,
            "key": API_KEY
        }
        response = requests.get(base_url, params=params)
        data = response.json()

        if 'results' in data:
            for result in data['results']:
                elevations.append(result['elevation'])
        else:
            elevations.extend([None] * len(chunk))

    return elevations


def distance(coord1, coord2):
    """Calculates the distance between two coordinates."""
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)


def nearest_neighbor_sort(coordinates, elevations):
    """Sorts coordinates and elevations using the nearest neighbor method."""
    if not coordinates:
        return [], []

    sorted_coords = [coordinates[0]]
    sorted_elevations = [elevations[0]]
    remaining_coords = coordinates[1:]
    remaining_elevations = elevations[1:]

    while remaining_coords:
        last_coord = sorted_coords[-1]
        distances = [distance(last_coord, coord) for coord in remaining_coords]
        nearest_index = np.argmin(distances)
        sorted_coords.append(remaining_coords.pop(nearest_index))
        sorted_elevations.append(remaining_elevations.pop(nearest_index))

    return sorted_coords, sorted_elevations
