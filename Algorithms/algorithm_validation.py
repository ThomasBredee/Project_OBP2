from Algorithms.solver_VRPy import vrpy_solver
from Algorithms.distance_calculator import RoadDistanceCalculator
from Algorithms.solver_pyvrp import VRPSolver
from Candidate_Ranking.Rankings import CandidateRanking
from Algorithms.algorithm_evaluation import AlgorithmEvaluation
import pandas as pd
import time


#######INPUTS FROM THE MODEL VARIABLES
TRUCK_CAPACITY =  40
CHOSEN_COMPANY = "Innovative Ventures"
CHOSEN_CANDIDATE = "Innovative Ventures"
LONG_DEPOT = 5.26860985
LAT_DEPOT = 52.2517788

if __name__ == "__main__":

    ###get the data
    input_df = pd.read_csv("Data/manyLarge.csv")
    # print(input_df.name.value_counts())
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


    # --- VRPy solver ---
    Solver = vrpy_solver(distance_matrix_vrp)
    m = Solver.solve(TRUCK_CAPACITY, 500, 2)
    # distance = Solver.returnDistance(m)[1]
    print("VRPy Solver")
    print("Routes")
    routes = Solver.returnRoutes(m)

    Solver.calculate_distance_per_order(routes)
    Solver.plotRoute(routes, input_df)

    vrp_solver = VRPSolver()

    # --- PYVRP Solver ---
    # print("Solving VRP for Collaboration...")
    print("PyVRP Solver")
    model_collab, current_names_collab = vrp_solver.build_model(
        input_df=input_df,
        chosen_company=CHOSEN_COMPANY,
        chosen_candidate=CHOSEN_CANDIDATE,
        distance_matrix=distance_matrix_vrp,
        truck_capacity=TRUCK_CAPACITY
    )
    print("Routes")
    solution_collab, routes_collab = vrp_solver.solve(
        m=model_collab,
        max_runtime=0.5,
        display=False,
        current_names=current_names_collab
    )

    total_distance_collab, avg_distance_per_order_collab = vrp_solver.calculate_distance_per_order(
        routes=routes_collab,
        distance_matrix=distance_matrix_vrp
    )

    vrp_solver.plotRoute(routes_collab, input_df)



