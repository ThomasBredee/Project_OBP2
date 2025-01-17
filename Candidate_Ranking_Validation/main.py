########################################################
#                                                       #
# Created on: 11/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
# Updated on: 16/01/2025                                #
# Updated by: Dennis Botman                             #
#                                                       #
#########################################################


from Algorithms.distance_calculator import RoadDistanceCalculator
from Algorithms.solver_pyvrp import VRPSolver
from Candidate_Ranking.Rankings import CandidateRanking
from Algorithms.algorithm_evaluation import AlgorithmEvaluation
import pandas as pd
import time


#######INPUTS FROM THE MODEL VARIABLES
TRUCK_CAPACITY =  2
CHOSEN_COMPANY = "Visionary Ventures"
CHOSEN_CANDIDATE = "NextGen Solutions"
LONG_DEPOT = 5.26860985
LAT_DEPOT = 52.2517788

if __name__ == "__main__":

    # Load data
    input_df = pd.read_csv("Data/mini.csv")
    input_df_orig = pd.read_csv("Data/mini.csv")
    input_df['name'] = input_df.groupby('name').cumcount().add(1).astype(str).radd(input_df['name'] + "_")
    print(input_df_orig['name'].unique())

    # Add depot to the input dataframe
    distance_calc = RoadDistanceCalculator()
    input_df_for_ranking = distance_calc.add_depot(input_df, LAT_DEPOT, LONG_DEPOT)

    # Iterate over truck capacities
    for truck_cap in [2, 4, 6, 8, 10]:
        print(f"Processing for truck capacity: {truck_cap}")

        # Initialize DataFrame to store average distances for this truck capacity
        solutions_df = pd.DataFrame(index=input_df_orig['name'].unique(), columns=input_df_orig['name'].unique())
        solutions_df = solutions_df.fillna('NoSolution')

        # Iterate over companies and candidates
        for chosen_company in solutions_df.columns:
            for chosen_candidate in solutions_df.index:
                if chosen_candidate == chosen_company:
                    solutions_df.at[chosen_candidate, chosen_company] = 9999999
                    continue

                # Instantiate distance calculator and VRP solver
                vrp_solver = VRPSolver()

                # Calculate the full distance matrix for the chosen company
                distance_matrix_ranking = distance_calc.calculate_distance_matrix(
                    input_df_for_ranking,
                    chosen_company=chosen_company,
                    candidate_name=None,
                    method="haversine",
                    computed_distances_df=None
                )

                # Calculate the distance matrix for VRP
                distance_matrix_vrp = distance_calc.calculate_distance_matrix(
                    input_df_for_ranking,
                    chosen_company=chosen_company,
                    candidate_name=chosen_candidate,
                    method="haversine",
                    computed_distances_df=distance_matrix_ranking
                )

                # Build and solve the VRP model
                model_collab, current_names_collab = vrp_solver.build_model(
                    input_df=input_df_for_ranking,
                    chosen_company=chosen_company,
                    chosen_candidate=chosen_candidate,
                    distance_matrix=distance_matrix_vrp,
                    truck_capacity=truck_cap
                )

                solution_collab, routes_collab = vrp_solver.solve(
                    m=model_collab,
                    max_runtime=4,
                    display=False,
                    current_names=current_names_collab
                )

                # Calculate total and average distances
                total_distance_collab, avg_distance_per_order_collab = vrp_solver.calculate_distance_per_order(
                    routes=routes_collab,
                    distance_matrix=distance_matrix_vrp
                )

                # Save the average distance in the solutions DataFrame
                if solution_collab:  # Check if the solution is feasible
                    solutions_df.at[chosen_candidate, chosen_company] = avg_distance_per_order_collab
                else:
                    solutions_df.at[chosen_candidate, chosen_company] = float(
                        'inf')  # Use a large value for no solution

        # Create a new DataFrame for ranks
        ranks_df = pd.DataFrame(index=solutions_df.index, columns=solutions_df.columns)

        # Assign ranks based on average distances
        for chosen_company in solutions_df.columns:
            # Extract distances for the current company
            distances = solutions_df[chosen_company]

            # Rank candidates based on distance (ascending order)
            ranked_values = distances.rank(method='dense', ascending=True)

            # Assign ranks to ranks_df, handling 'NoSolution'
            ranks_df[chosen_company] = distances.apply(
                lambda x: 'NoSolution' if x == float('inf') else int(ranked_values[distances == x].iloc[0])
            )

        # Save results for this truck capacity to an Excel file
        file_name = f"ranking_results_truck_cap_{truck_cap}.xlsx"
        with pd.ExcelWriter(file_name) as writer:
            solutions_df.to_excel(writer, sheet_name='Solutions (Avg Distances)')
            ranks_df.to_excel(writer, sheet_name='Ranks')

        print(f"Results saved for truck capacity {truck_cap} to {file_name}")
