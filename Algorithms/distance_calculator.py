#########################################################
#                                                       #
# Created on: 11/01/2024                                #
# Created by: Dennis Botman                             #
#                                                       #
#########################################################


import pandas as pd
import requests
import math
from concurrent.futures import ThreadPoolExecutor


class RoadDistanceCalculator:
    def __init__(self, osrm_url="http://localhost:5000"):
        self.osrm_url = osrm_url
        self.session = requests.Session()

    def calculate_distance_osrm(self, start, end):
        """Calculate the road distance between two points using the OSRM server."""
        url = f"{self.osrm_url}/route/v1/driving/{start[1]},{start[0]};{end[1]},{end[0]}"
        params = {"overview": "false"}
        response = self.session.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return data["routes"][0]["distance"] / 1000  # Convert meters to kilometers
        else:
            response.raise_for_status()

    def calculate_distance_haversine(self, start, end):
        """Calculate the Haversine distance between two points."""
        R = 6371  # Radius of the Earth in kilometers
        lat1, lon1 = start
        lat2, lon2 = end
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def calculate_distance_matrix(self, points, filter_string=None, flavor="osrm"):
        """
        Calculate a distance matrix for a filtered list of points.

        Args:
            points (pd.DataFrame): A DataFrame containing 'name', 'lat', and 'lon' columns.
            filter_string (str): A substring to filter rows by company name.
            flavor (str): The method to calculate distances ('osrm' or 'haversine').

        Returns:
            pd.DataFrame: A DataFrame with distances between all filtered rows and all columns.
        """
        # Filter rows based on the substring
        if filter_string:
            filtered_points = points[points["name"].str.contains(filter_string, case=False)]
        else:
            filtered_points = points

        row_names = filtered_points["name"].tolist()
        row_lats = filtered_points["lat"].tolist()
        row_lons = filtered_points["lon"].tolist()

        column_names = points["name"].tolist()
        column_lats = points["lat"].tolist()
        column_lons = points["lon"].tolist()

        # Initialize the matrix
        matrix = pd.DataFrame(index=row_names, columns=column_names, dtype=float)

        # Calculate distances for filtered rows (rows x all columns)
        pairs = [
            ((row_lats[i], row_lons[i]), (column_lats[j], column_lons[j]))
            for i in range(len(row_lats))
            for j in range(len(column_lats))
        ]

        def calculate_pair(pair):
            start, end = pair
            if flavor == "osrm":
                return self.calculate_distance_osrm(start, end)
            elif flavor == "haversine":
                return self.calculate_distance_haversine(start, end)
            else:
                raise ValueError("Invalid flavor specified. Use 'osrm' or 'haversine'.")

        with ThreadPoolExecutor(max_workers=10) as executor:
            distances = list(executor.map(calculate_pair, pairs))

        # Fill the matrix
        index = 0
        for i in range(len(row_lats)):
            for j in range(len(column_lats)):
                matrix.iloc[i, j] = distances[index]
                index += 1

        return matrix
