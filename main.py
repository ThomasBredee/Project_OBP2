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
    heuristic = Heuristics()

    dashboard.printInput(dashboard.input_df)

    if dashboard.sidebarButton() and dashboard.input_df is not None:   # DIT ER NOG LATER IN VERWERKEN
        dashboard.Test()

        euclidean_distance_matrix = algorithm.calculate_distance_matrix(dashboard.input_df)



        if dashboard.heuristics_choice == "greedy":
            ranking = heuristic.greedy(euclidean_distance_matrix)
            dashboard.Test()
        if dashboard.heuristics_choice == "bounding_box":
            ranking = heuristic.bounding_box(dashboard.input_df, euclidean_distance_matrix)


