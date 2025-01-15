from osmnx.distance import euclidean

from Algorithms.distance_calculator import RoadDistanceCalculator
from Dashboard.dashboard import Dashboard
from Heuristics.Rankings import Heuristics
import pandas as pd
import time

if __name__ == "__main__":
    euclidean_distance_matrix = None
    dashboard = Dashboard()
    algorithm = RoadDistanceCalculator()

    if dashboard.input_df is not None:
        euclidean_distance_matrix = algorithm.calculate_distance_matrix(dashboard.input_df)
        dashboard.Test()

    if dashboard.execute_Ranking and euclidean_distance_matrix is not None:
        heuristic = Heuristics()
        if dashboard.heuristics_choice == "greedy":
            dashboard.Test()
             #### MOET HIER DE DISTANCE MATRIX MEEGEGEVEN WORDEN? ####
            ranking = heuristic.greedy(euclidean_distance_matrix)
        if dashboard.heuristics_choice == "boundingbox":
            #### EN MOET HIER DAN DE DISTANCE MATRIX, EN DE INPUT_FILE MEEGEGEVEN WORDEN ####
            ranking = heuristic.bounding_box(dashboard.input_df, euclidean_distance_matrix)
