#########################################################
#                                                       #
# Created on: 11/01/2024                                #
# Created by: Dennis Botman                             #
#                                                       #
#########################################################

from Algorithms.distance_calculator import RoadDistanceCalculator
from Algorithms.solver_pyvrp import VRPSolver
import pandas as pd
import time

if __name__ == "__main__":

    input_df = pd.read_csv("Data/manyLarge.csv")
    # print(input_df.name.value_counts())

    input_df['name'] = input_df.groupby('name').cumcount().add(1).astype(str).radd(input_df['name'] + "_")

    filter_comp1 = "Innovative Ventures"
    filter_comp2 = "Global Enterprises"

    #Get Distance matrix
    start_time = time.time()

    vrp_solver = VRPSolver()
    input_df = vrp_solver.addDepot(input_df, input_df.lat.mean(), input_df.lon.mean())

    calculator = RoadDistanceCalculator()
    distance_matrix = calculator.calculate_distance_matrix(

        input_df, filter_comp1=filter_comp1, filter_comp2=filter_comp2, flavor="haversine"
    )


    model, current_names = vrp_solver.buildModel(input_df, filter_comp1, filter_comp2, distance_matrix)

    solution, route = vrp_solver.solve(model, max_runtime=1, display=False, current_names=current_names)

    route_distance = vrp_solver.validateRoute(solution, route, distance_matrix, input_df)

    print("Time: ", time.time()-start_time)


