from Heuristics.Rankings import Heuristics
from Algorithms.distance_calculator import RoadDistanceCalculator
import pandas as pd
import time


if __name__ == "__main__":
    input_df = pd.read_csv("../Data/manyLarge.csv")

    #Get Distance matrix
    start_time = time.time()
    calculator = RoadDistanceCalculator()
    distance_matrix = calculator.calculate_distance_matrix(
        input_df, filter_string="Pioneer Networks", flavor="haversine"
    )
    algorithms = Candidate_Rankings()
    algorithm1 = algorithms.greedy(distance_matrix)

    #print(time.time()-start_time)

