#########################################################
#                                                       #
# Created on: 11/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
# Updated on: 21/01/2025                                #
# Updated by: Dennis Botman                             #
#                                                       #
#########################################################

import pandas as pd
import requests
import math
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np

class RoadDistanceCalculator:
    def __init__(self, osrm_url="http://localhost:5000"):
        self.osrm_url = osrm_url
        self.session = requests.Session()

    def _calculate_distance_osrm(self, start, end, retries=6, backoff_factor=10):
        """Calculate road distance between two points using osrm with retries."""
        for attempt in range(retries):
            try:
                url_ab = f"{self.osrm_url}/route/v1/driving/{start[1]},{start[0]};{end[1]},{end[0]}"
                url_ba = f"{self.osrm_url}/route/v1/driving/{end[1]},{end[0]};{start[1]},{start[0]}"
                response_ab = self.session.get(url_ab, params={"overview": "false"}, timeout=10)
                response_ba = self.session.get(url_ba, params={"overview": "false"}, timeout=10)

                if response_ab.status_code == 200 and response_ba.status_code == 200:
                    distance_ab = response_ab.json()["routes"][0]["distance"] / 1000  # Convert to km
                    distance_ba = response_ba.json()["routes"][0]["distance"] / 1000  # Convert to km
                    return distance_ab, distance_ba  # Return both distances

            except ConnectionError:
                print(f"Connection failed. Retrying {attempt + 1}/{retries}...")
                time.sleep(120)

        raise ConnectionError("Failed to connect to osrm after retries.")

    def _calculate_distance_haversine(self, start, end):
        """Calculate the Haversine distance between two points (using haversine formula)"""

        r = 6371  # Radius of the Earth in kilometers
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
        return r * c

    def calculate_distance_matrix(self, locations_df, chosen_company=None, candidate_name=None, method="osrm",
                                  computed_distances_df=None):
        """
        Calculate a distance matrix for selected locations.

        Args:
            locations_df (pd.DataFrame): A DataFrame containing 'name', 'lat', and 'lon' columns.
            chosen_company (str): The company for which potential partnerships are being evaluated.
            candidate_name (str): The candidate company for distance calculations.
            method (str): The method to calculate distances ('osrm' or 'haversine').
            computed_distances_df (pd.DataFrame): Previously calculated distances, if available.

        Returns:
            pd.DataFrame: A DataFrame with asymmetric distances between locations.
        """
        if computed_distances_df is None:
            # Filter only rows belonging to chosen_company
            filtered_rows_df = locations_df[locations_df["name"].str.contains(chosen_company, case=False, na=False)]
            all_columns_df = locations_df

            row_names = filtered_rows_df["name"].tolist()
            row_lats = filtered_rows_df["lat"].tolist()
            row_longs = filtered_rows_df["lon"].tolist()

            column_names = all_columns_df["name"].tolist()
            column_lats = all_columns_df["lat"].tolist()
            column_longs = all_columns_df["lon"].tolist()

            # Initialize matrix with chosen company rows and all columns
            matrix = pd.DataFrame(float('nan'), index=row_names, columns=column_names)

            pairs = [
                ((row_lats[i], row_longs[i]), (column_lats[j], column_longs[j]))
                for i in range(len(row_lats))
                for j in range(len(column_lats))
            ]

            def calculate_pair(pair):
                (start_lat, start_lon), (end_lat, end_lon) = pair
                start = (start_lat, start_lon)
                end = (end_lat, end_lon)

                if method == "osrm":
                    distance_ab, distance_ba = self._calculate_distance_osrm(start, end)
                elif method == "haversine":
                    distance_ab = self._calculate_distance_haversine(start, end)
                    distance_ba = self._calculate_distance_haversine(end, start)
                else:
                    raise ValueError("Invalid method specified. Use 'osrm' or 'haversine'.")

                return distance_ab, distance_ba

            # Use threading to speed up calculations
            with ThreadPoolExecutor(max_workers=2) as executor:
                distances = list(executor.map(calculate_pair, pairs))

            index = 0
            for i, row_name in enumerate(row_names):
                for j, col_name in enumerate(column_names):
                    distance_ab, distance_ba = distances[index]
                    matrix.at[row_name, col_name] = distance_ab  # A to B
                    index += 1

        else:
            # Filter the computed_distances_df to include only chosen_company, candidate_name, and 'Depot'
            filtered_columns = [col for col in computed_distances_df.columns if
                                chosen_company in col or candidate_name in col or 'Depot' in col]
            updated_matrix = computed_distances_df.loc[:, filtered_columns].copy()

            # Add rows for candidate_name and Depot
            candidate_df = locations_df[locations_df["name"].str.contains(candidate_name, case=False, na=False) |
                                        locations_df["name"].str.contains("Depot", case=False, na=False)]
            candidate_names = candidate_df["name"].tolist()
            candidate_lats = candidate_df["lat"].tolist()
            candidate_longs = candidate_df["lon"].tolist()

            for candidate_name, candidate_lat, candidate_lon in zip(candidate_names, candidate_lats, candidate_longs):
                if candidate_name not in updated_matrix.index:
                    updated_matrix.loc[candidate_name] = float('nan')

                for col_name in updated_matrix.columns:
                    if pd.isna(updated_matrix.loc[candidate_name, col_name]):
                        col_lat = locations_df.loc[locations_df["name"] == col_name, "lat"].values[0]
                        col_lon = locations_df.loc[locations_df["name"] == col_name, "lon"].values[0]

                        start = (candidate_lat, candidate_lon)
                        end = (col_lat, col_lon)

                        if method == "osrm":
                            distance_ab, distance_ba = self._calculate_distance_osrm(start, end)
                        elif method == "haversine":
                            distance_ab = self._calculate_distance_haversine(start, end)
                            distance_ba = self._calculate_distance_haversine(end, start)

                        updated_matrix.loc[candidate_name, col_name] = distance_ab
                        updated_matrix.loc[col_name, candidate_name] = distance_ba

            updated_matrix.loc["Depot", "Depot"] = 0

            return updated_matrix

        return matrix

    def add_depot(self, input_df, depot_lat, depot_lon):
        depot_row = {"name": "Depot", 'lat': depot_lat, 'lon': depot_lon}
        return pd.concat([pd.DataFrame([depot_row]), input_df], ignore_index=True)

    def calculate_square_matrix(self, df):
        # Extract latitudes and longitudes as NumPy arrays
        latitudes = df['lat'].to_numpy()
        longitudes = df['lon'].to_numpy()

        # Convert to radians for vectorized haversine computation
        lat_radians = np.radians(latitudes)
        lon_radians = np.radians(longitudes)

        # Compute pairwise differences
        lat_diff = lat_radians[:, None] - lat_radians[None, :]
        lon_diff = lon_radians[:, None] - lon_radians[None, :]

        # Compute haversine components
        a = (
                np.sin(lat_diff / 2) ** 2 +
                np.cos(lat_radians[:, None]) * np.cos(lat_radians[None, :]) * np.sin(lon_diff / 2) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        # Earth's radius in kilometers (adjust if needed)
        earth_radius_km = 6371.0
        distance_matrix = earth_radius_km * c

        # Convert to DataFrame and set names
        distance_matrix = pd.DataFrame(distance_matrix, index=df['name'], columns=df['name'])

        return distance_matrix


