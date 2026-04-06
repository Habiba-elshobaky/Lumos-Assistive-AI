from routingpy import ORS
import json
import os
from keys import ORS_API_KEY

client = ORS(api_key=ORS_API_KEY)
CACHE_FILE = "route_cache.json"

def get_navigation_data(start_coords, end_coords):
    """ Fetches walking directions using OpenRouteService """
    try:
        # ORS uses [Longitude, Latitude]
        route = client.directions(locations=[start_coords, end_coords], profile='foot-walking')
        steps = route.raw['features'][0]['properties']['segments'][0]['steps']
        
        with open(CACHE_FILE, 'w') as f:
            json.dump(steps, f)
        return steps
    except Exception as e:
        print(f">>> [ORS ERROR]: {e}")
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f: return json.load(f)
        return []