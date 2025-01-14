from Algorithms.distance_calculator import RoadDistanceCalculator
from Dashboard.dashboard import Dashboard
import pandas as pd
import time

if __name__ == "__main__":
    dashboard = Dashboard()

    input_df = None

    #input_df = pd.read_csv("Data\\manyLarge.csv")

    #Get Distance matrix
    start_time = time.time()
    #calculator = RoadDistanceCalculator()
    #distance_matrix = calculator.calculate_distance_matrix(
    #    input_df, filter_string="Visionary Ventures", flavor="haversine"
    #)
    while input_df is None:
        time.sleep(1)

    input_df = dashboard.df

    if input_df is not None:
        input_df['name'] = input_df.groupby('name').cumcount().add(1).astype(str).radd(input_df['name']+"_")

        route = [
            "Pioneer Networks_1", "Pioneer Networks_3", "NextGen Technologies_1",
            "NextGen Technologies_5", "Pioneer Networks_5", "Pioneer Networks_2"
        ]

        dashboard.showMap(input_df, route)

    print(time.time()-start_time)