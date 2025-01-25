#########################################################
#                                                       #
# Created on: 25/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
#########################################################

import pandas as pd
import numpy as np
from Candidate_Ranking.ranking_methods import CandidateRanking
from Algorithms.solver_pyvrp import VRPSolver
import random

class DataFramePreparer:
    def __init__(self):
        # Initialize any necessary attributes here (if any)
        pass

    def clean_and_split_data(self, dataframes):
        # Combine all dataframes, preserving duplicates within the same DataFrame
        seen_names = set()
        filtered_dfs = []
        for df in dataframes:
            df_filtered = df[~df['name'].isin(seen_names)]  # Keep only unique names across DataFrames
            seen_names.update(df['name'].unique())  # Add current names to the set
            filtered_dfs.append(df_filtered)

        # Make 1 main df
        input_df = pd.concat(filtered_dfs, ignore_index=True)

        # Split data to make sure that the test set is never touched and can be predicted to measure overfitting in the models
        unique_companies = input_df['name'].unique()

        # Shuffle the array of unique companies
        np.random.shuffle(unique_companies)

        # Calculate the split index
        split_idx = int(len(unique_companies) * 0.8)

        # Split companies into training and testing
        train_companies = unique_companies[:split_idx]
        test_companies = unique_companies[split_idx:]

        # Create the training and testing DataFrame
        self.training_df = input_df[input_df['name'].isin(train_companies)]
        self.testing_df = input_df[input_df['name'].isin(test_companies)]

        return self.training_df, self.testing_df

    def get_features_greedy(self, df_modified, sliced_distance_matrix_ranking, sliced_distance_matrix_vrp, max_truck_cap, chosen_company, chosen_candidate):
        #Get random truck capacity
        truck_cap = random.choice(list(range(2, max_truck_cap+1, 1)))

        #Get total greedy distance
        ranking = CandidateRanking()
        predicted_ranking_greedy = ranking.greedy(sliced_distance_matrix_ranking,chosen_company)
        total_greedy_dist = predicted_ranking_greedy.loc[chosen_candidate][0]

        #Get total route lenght based on distance matrix by summing sequential movements
        route_sequence = sliced_distance_matrix_vrp.index.tolist()
        total_route_distance = sum(
            sliced_distance_matrix_vrp.loc[route_sequence[i], route_sequence[i + 1]]
            for i in range(len(route_sequence) - 1)
        )

        #Get the number of locations
        num_stops = len(sliced_distance_matrix_vrp.columns) - 1

        #Calculate minimum and maximum distances from the depot
        depot_distances = sliced_distance_matrix_vrp.loc['Depot'].copy().drop('Depot')  # Drop self-distance
        min_distance = depot_distances.min()
        max_distance = depot_distances.max()

        #Solve the vrp to get real distance per order
        vrp_solver = VRPSolver()
        model_collab, current_names_collab = vrp_solver.build_model(
            input_df=df_modified,
            chosen_company=chosen_company,
            chosen_candidate=chosen_candidate,
            distance_matrix=sliced_distance_matrix_vrp,
            truck_capacity=truck_cap
        )
        solution_collab, routes_collab = vrp_solver.solve(
            model=model_collab,
            max_runtime=1,
            display=False,
            current_names=current_names_collab
        )

        total_distance_collab, avg_distance_per_order_collab = vrp_solver.calculate_distance_per_order(
            routes=routes_collab,
            distance_matrix=sliced_distance_matrix_vrp
        )

        #Append results to list
        results_row = {
            "totalsum": total_route_distance,
            "number_stops": num_stops,
            "Max_to_depot": max_distance,
            "Min_to_depot": min_distance,
            "vehicle_cap": truck_cap,
            "greedy_total_sum": total_greedy_dist,
            "real_km_order": avg_distance_per_order_collab,
            "chosen_company": chosen_company,
            "chosen_candidate": chosen_candidate
        }

        return results_row





    # # Simple retry logic in place when calling the function
    # max_retries = 6
    # wait_time = 30  # 60 seconds before retrying
    #
    # for attempt in range(max_retries):
    #     try:
    #         print(f"Attempt {attempt + 1} of {max_retries} to calculate the distance matrix...")
    #         distance_matrix_ranking = distance_calc.calculate_distance_matrix(
    #             input_df,
    #             chosen_company=chosen_company,
    #             candidate_name=None,
    #             method="osrm",
    #             computed_distances_df=None
    #         )
    #         print("Distance matrix calculated successfully.")
    #         break  # If successful, exit the retry loop
    #     except ConnectionError as e:
    #         print(f"Connection failed: {e}")
    #         if attempt < max_retries - 1:
    #             print(f"Retrying in {wait_time} seconds...")
    #             time.sleep(wait_time)
    #         else:
    #             print("Max retries reached. Failed to calculate distance matrix.")
    #             distance_matrix_ranking = None  # Ensure a value is assigned in case of failure
    # else:
    #     print("Error: No distance matrix was calculated.")

    # Data collection for results