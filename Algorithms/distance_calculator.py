import requests
import pandas as pd
import math

OSRM_API_URL = "http://router.project-osrm.org"

class RouteCalculator:
    def __init__(self, osrm_url: str = OSRM_API_URL):
        self.osrm_url = osrm_url

    def calculate_route(self, start: tuple, end: tuple) -> float:
        """Calculate the real route distance between two coordinates using the OSRM API.
        Args:
            start (tuple): A tuple of (latitude, longitude) for the starting point.
            end (tuple): A tuple of (latitude, longitude) for the ending point.
        Returns:
            float: Distance in kilometers.
        """
        url = f"{self.osrm_url}/route/v1/driving/{start[1]},{start[0]};{end[1]},{end[0]}"
        params = {'overview': 'false'}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return data['routes'][0]['distance'] / 1000  # Convert meters to kilometers
        else:
            response.raise_for_status()

    def calculate_haversine(self, start: tuple, end: tuple) -> float:
        """Calculate the Haversine distance between two coordinates.
        Args:
            start (tuple): A tuple of (latitude, longitude) for the starting point.
            end (tuple): A tuple of (latitude, longitude) for the ending point.
        Returns:
            float: Distance in kilometers.
        """
        R = 6371  # Radius of the Earth in kilometers
        lat1, lon1 = start
        lat2, lon2 = end
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def calculate_routes_from_dataframe(self, df: pd.DataFrame, method: str = 'osrm') -> pd.DataFrame:
        """Calculate distances for all pairs of coordinates in a DataFrame using the specified method.
        Args:
            df (pd.DataFrame): A DataFrame containing 'name', 'lat', and 'lon' columns.
            method (str): The method to use ('osrm' or 'haversine').
        Returns:
            pd.DataFrame: A DataFrame with distances between all pairs.
        """
        num_locations = len(df)
        distances = pd.DataFrame(index=df['name'], columns=df['name'], dtype=float)

        for i, (name1, lat1, lon1) in df.iterrows():
            for j, (name2, lat2, lon2) in df.iterrows():
                if i == j:
                    distances.loc[name1, name2] = 0
                else:
                    start = (lat1, lon1)
                    end = (lat2, lon2)
                    if method == 'osrm':
                        distances.loc[name1, name2] = self.calculate_route(start, end)
                    elif method == 'haversine':
                        distances.loc[name1, name2] = self.calculate_haversine(start, end)
                    else:
                        raise ValueError("Invalid method. Use 'osrm' or 'haversine'.")

        return distances
