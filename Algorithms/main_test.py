#########################################################
#                                                       #
# Created on: 11/01/2024                                #
# Created by: Dennis Botman                             #
#                                                       #
#########################################################

from Algorithms.distance_calculator import RoadDistanceCalculator
import pandas as pd
import time




if __name__ == "__main__":
    input_df = pd.read_csv("Data\\manyLarge.csv")

    #Get Distance matrix
    start_time = time.time()
    calculator = RoadDistanceCalculator()
    distance_matrix = calculator.calculate_distance_matrix(
        input_df, filter_string="Visionary Ventures", flavor="osrm"
    )
    print(time.time()-start_time)





