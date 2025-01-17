#########################################################
#                                                       #
# Created on: 11/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
# Updated on: 16/01/2025                                #
# Updated by: Dennis Botman                             #
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

    def _calculate_distance_osrm(self, start, end):
        """Calculate the road distance between two points using the OSRM local server"""

        url = f"{self.osrm_url}/route/v1/driving/{start[1]},{start[0]};{end[1]},{end[0]}"
        params = {"overview": "false"}
        response = self.session.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return data["routes"][0]["distance"] / 1000  # Convert to kilometers
        else:
            response.raise_for_status()

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

    def calculate_distance_matrix(self, locations_df, chosen_company=None, candidate_name=None, method="osrm", computed_distances_df=None):
        """
        Calculate a distance matrix for a filtered list of begin and end points

        Args:
            locations_df (pd.DataFrame): A DataFrame containing 'name', 'lat', and 'lon' columns.
            chosen_company (str): This is the company where the manager wants to see potential partnerships for.
            candidate_name (str): The candidate company to add new locations for distance calculation.
            method (str): The method to calculate distances ('osrm' or 'haversine').
            computed_distances_df (pd.DataFrame): A DataFrame of already calculated distances.

        Returns:
            pd.DataFrame: A DataFrame with distances between all filtered rows and all columns.
        """
        if computed_distances_df is None:
            # Get rows for the chosen company and all columns for other locations
            filtered_rows_df = locations_df[locations_df["name"].str.contains(chosen_company, case=False, na=False)]
            all_columns_df = locations_df

            row_names = filtered_rows_df["name"].tolist()
            row_lats = filtered_rows_df["lat"].tolist()
            row_longs = filtered_rows_df["lon"].tolist()

            column_names = all_columns_df["name"].tolist()
            column_lats = all_columns_df["lat"].tolist()
            column_longs = all_columns_df["lon"].tolist()

            matrix = pd.DataFrame(index=row_names, columns=column_names, dtype=float)

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
                    return self._calculate_distance_osrm(start, end)
                elif method == "haversine":
                    return self._calculate_distance_haversine(start, end)
                else:
                    raise ValueError("Invalid method specified. Use 'osrm' or 'haversine'.")

            with ThreadPoolExecutor(max_workers=10) as executor:
                distances = list(executor.map(calculate_pair, pairs))

            index = 0
            for i in range(len(row_lats)):
                for j in range(len(column_lats)):
                    matrix.iloc[i, j] = distances[index]
                    index += 1

        else:
            # Filter the computed_distances_df to only keep columns containing chosen_company, candidate_name, or 'Depot'
            filtered_columns = [col for col in computed_distances_df.columns if
                                chosen_company in col or candidate_name in col or 'Depot' in col]
            updated_matrix = computed_distances_df.loc[:, filtered_columns].copy()

            # Add rows for each location in the candidate_name
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
                            distance = self._calculate_distance_osrm(start, end)
                        elif method == "haversine":
                            distance = self._calculate_distance_haversine(start, end)

                        updated_matrix.loc[candidate_name, col_name] = distance
                        updated_matrix.loc[col_name, candidate_name] = distance
                updated_matrix.loc["Depot", "Depot"] = 0
            return updated_matrix

        return matrix

    def add_depot(self, input_df, depot_lat, depot_lon):
        depot_row = {"name": "Depot", 'lat': depot_lat, 'lon': depot_lon}
        return pd.concat([pd.DataFrame([depot_row]), input_df], ignore_index=True)

