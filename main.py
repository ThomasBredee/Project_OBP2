from Algorithms.distance_calculator import RoadDistanceCalculator
from Dashboard.dashboard import Dashboard
from Heuristics.Rankings import Heuristics
import pandas as pd
import time

if __name__ == "__main__":

    dashboard = Dashboard()


    if dashboard.execute_Ranking:
        heuristic = Heuristics()
        #if dashboard.heuristics_choice == "greedy":
        #    ranking = heuristic.greedy()
        #if dashboard.heuristics_choice == "boundingbox":
        #    ranking = heuristic.boundingbox()

        ranking = None