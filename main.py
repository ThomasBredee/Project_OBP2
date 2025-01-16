from Algorithms.distance_calculator import RoadDistanceCalculator
from Dashboard.dashboard import Dashboard
from Candidate_Ranking.Rankings import CandidateRanking
from Algorithms.solver_pyvrp import VRPSolver
import streamlit as st

LONG_DEPOT = 5.26860985
LAT_DEPOT = 52.2517788

if __name__ == "__main__":
    dashboard = Dashboard()
    algorithm = RoadDistanceCalculator()
    heuristic = CandidateRanking()

    dashboard.input_df['name'] = dashboard.input_df.groupby('name').cumcount().add(1).astype(str).radd(
    dashboard.input_df['name'] + "_")
    # input_df_dashes = dashboard.input_df.copy()

    # Ensure stateful variables are accessed correctly
    if st.session_state.execute_Ranking and dashboard.input_df is not None:
        euclidean_distance_matrix = algorithm.calculate_distance_matrix(dashboard.input_df, dashboard.company_1, method="haversine")

        if dashboard.heuristics_choice == "greedy":
            ranking = heuristic.greedy(euclidean_distance_matrix)
            dashboard.candidate_ranking = ranking
            dashboard.choose_candidate(ranking)
        if dashboard.heuristics_choice == "bounding_box":
            ranking = heuristic.bounding_box(dashboard.input_df, euclidean_distance_matrix)
            dashboard.choose_candidate(ranking)

        # Check if VRP execution is triggered
        if st.session_state.execute_VRP and dashboard.company_2 is not None:
            input_df = algorithm.add_depot(dashboard.input_df, LAT_DEPOT, LONG_DEPOT)
            distance_matrix_vrp = algorithm.calculate_distance_matrix(
                input_df,
                chosen_company=dashboard.company_1,
                candidate_name=dashboard.company_2,
                method="haversine",
                computed_distances_df=euclidean_distance_matrix,
            )

            # Solve the VRP
            vrp_solver = VRPSolver()
            model, current_names = vrp_solver.build_model(
                input_df,
                dashboard.company_1,
                dashboard.company_2,
                distance_matrix_vrp,
                dashboard.vehicle_capacity,
            )

            solution, route = vrp_solver.solve(model, max_runtime=1, display=False, current_names=current_names)
            print(solution)
            dashboard.Test()
            #TEST
