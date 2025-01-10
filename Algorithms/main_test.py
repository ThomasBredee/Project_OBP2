from Algorithms.distance_calculator import RouteCalculator
import pandas as pd

df = pd.read_csv('Algorithms/mini.csv')
rc = RouteCalculator()
osrm_distances = rc.calculate_routes_from_dataframe(df, method='osrm')
haversine_matrix = dm.create_distance_matrix(method='haversine')
osrm_matrix = dm.create_distance_matrix(method='osrm')

print(len(df))
print(haversine_matrix.shape)

# Example usage:
# df = pd.DataFrame({
#     'name': ['Location1', 'Location2'],
#     'lat': [52.3676, 52.0907],
#     'lon': [4.9041, 5.1214]
# })
# rc = RouteCalculator()
# osrm_distances = rc.calculate_routes_from_dataframe(df, method='osrm')
# haversine_distances = rc.calculate_routes_from_dataframe(df, method='haversine')
# print("OSRM Distances:", osrm_distances)
# print("Haversine Distances:", haversine_distances)