#########################################################
#                                                       #
# Created on: 11/01/2024                                #
# Created by: Dennis Botman                             #
#                                                       #
#########################################################

from Algorithms.distance_calculator import RoadDistanceCalculator
from Algorithms.solver_pyvrp import VRPSolver
from Candidate_Ranking.Rankings import CandidateRanking
import pandas as pd
import time


#######INPUTS FROM THE MODEL VARIABLES
TRUCK_CAPACITY =  5
CHOSEN_COMPANY = "Pioneer Networks"
CHOSEN_CANDIDATE = "NextGen Technologies"
LONG_DEPOT = 5.26860985
LAT_DEPOT = 52.2517788

if __name__ == "__main__":

    ###get the data
    input_df = pd.read_csv("Data/mini.csv")
    input_df['name'] = input_df.groupby('name').cumcount().add(1).astype(str).radd(input_df['name'] + "_")

    ###get distance matrix for chosen company
    distance_calc = RoadDistanceCalculator()
    distance_matrix_ranking = distance_calc.calculate_distance_matrix(input_df, chosen_company=CHOSEN_COMPANY,
        candidate_name=None, method="haversine", computed_distances_df=None)

    ###get candidate ranking
    ranking = CandidateRanking()
    algorithm1 = ranking.greedy(distance_matrix_ranking)

    ###get the full distance matrix of best company
    input_df = distance_calc.add_depot(input_df, LAT_DEPOT, LONG_DEPOT)
    distance_matrix_vrp= distance_calc.calculate_distance_matrix(input_df, chosen_company=CHOSEN_COMPANY,
        candidate_name=CHOSEN_CANDIDATE, method="haversine", computed_distances_df=distance_matrix_ranking)

    ###get best route
    vrp_solver = VRPSolver()
    model, current_names = vrp_solver.build_model(input_df, CHOSEN_COMPANY, CHOSEN_CANDIDATE, distance_matrix_vrp, TRUCK_CAPACITY)

    solution, route = vrp_solver.solve(model, max_runtime=1, display=False, current_names=current_names)

    route_distance = vrp_solver.validateRoute(solution, route, distance_matrix_vrp, input_df)


