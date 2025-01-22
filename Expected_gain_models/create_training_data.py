#########################################################
#                                                       #
# Created on: 20/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
# Created on: 21/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
#########################################################

from Algorithms.distance_calculator import RoadDistanceCalculator
from Algorithms.solver_pyvrp import VRPSolver
from Candidate_Ranking.Rankings import CandidateRanking
import pandas as pd
import random
import re
import time
import pandas as pd
from requests.exceptions import ConnectionError

####### INPUTS FROM THE MODEL VARIABLES
LONG_DEPOT = 5.26860985
LAT_DEPOT = 52.2517788

if __name__ == "__main__":

    # Load data
    # Load the data
    input_df1 = pd.read_csv("Data/mini.csv")
    input_df2 = pd.read_csv("Data/medium.csv")
    input_df3 = pd.read_csv("Data/many.csv")
    input_df4 = pd.read_csv("Data/manyLarge.csv")

    # Combine all dataframes, preserving duplicates within the same DataFrame
    dfs = [input_df1, input_df2, input_df3, input_df4]

    # Track unique 'name' values encountered across all DataFrames
    seen_names = set()

    # Create a list to store DataFrames after filtering out cross-duplicates
    filtered_dfs = []

    for df in dfs:
        df_filtered = df[~df['name'].isin(seen_names)]  # Keep only unique names across DataFrames
        seen_names.update(df['name'].unique())  # Add current names to the set
        filtered_dfs.append(df_filtered)

    # Concatenate the filtered DataFrames into a final combined DataFrame
    input_df = pd.concat(filtered_dfs, ignore_index=True)


    input_df['name'] = input_df.groupby('name').cumcount().add(1).astype(str).radd(input_df['name'] + "_")
    unique_companies = seen_names
    truck_caps = list(range(2, 20, 1))

    # Data collection for results
    results = []

    # Add depot to the input dataframe
    for row in range(0, 200):
        print('Making row:', row)
        distance_calc = RoadDistanceCalculator()

        # Convert unique_companies to a list before choosing
        unique_companies_list = list(unique_companies)
        chosen_company = random.choice(unique_companies_list)

        # Create a new list excluding the chosen company
        temp_list = [company for company in unique_companies_list if company != chosen_company]

        # Select another company from the remaining list
        chosen_candidate = random.choice(temp_list)

        # Random truck capacity selection
        truck_cap = random.choice(truck_caps)

        # Instantiate VRP solver
        vrp_solver = VRPSolver()

        # Simple retry logic in place when calling the function
        max_retries = 6
        wait_time = 30  # 60 seconds before retrying

        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1} of {max_retries} to calculate the distance matrix...")
                distance_matrix_ranking = distance_calc.calculate_distance_matrix(
                    input_df,
                    chosen_company=chosen_company,
                    candidate_name=None,
                    method="osrm",
                    computed_distances_df=None
                )
                print("Distance matrix calculated successfully.")
                break  # If successful, exit the retry loop
            except ConnectionError as e:
                print(f"Connection failed: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("Max retries reached. Failed to calculate distance matrix.")
                    distance_matrix_ranking = None  # Ensure a value is assigned in case of failure
        else:
            print("Error: No distance matrix was calculated.")

        # Calculate the full distance matrix for the chosen company

        ### Get candidate ranking
        ranking = CandidateRanking()
        algorithm1 = ranking.greedy(distance_matrix_ranking)

        total_greedy_dist = algorithm1.loc[chosen_candidate][0]

        # Calculate the distance matrix for VRP
        input_df_for_vrp = distance_calc.add_depot(input_df, LAT_DEPOT, LONG_DEPOT)
        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1} of {max_retries} to calculate the distance matrix...")
                distance_matrix_vrp = distance_calc.calculate_distance_matrix(
                    input_df_for_vrp,
                    chosen_company=chosen_company,
                    candidate_name=chosen_candidate,
                    method="osrm",
                    computed_distances_df=distance_matrix_ranking
                )
                print("Distance matrix calculated successfully.")
                break  # If successful, exit the retry loop
            except ConnectionError as e:
                print(f"Connection failed: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("Max retries reached. Failed to calculate distance matrix.")
                    distance_matrix_vrp = None  # Ensure a value is assigned in case of failure
        else:
            print("Error: No distance matrix was calculated.")

        # Filter the distance matrix
        filtered_matrix = distance_matrix_vrp

        # Create route sequence based on filtered columns order
        route_sequence = filtered_matrix.index.tolist()

        # Calculate total route distance by summing sequential movements
        total_route_distance = sum(
            filtered_matrix.loc[route_sequence[i], route_sequence[i + 1]]
            for i in range(len(route_sequence) - 1)
        )

        # Calculate the number of stops
        num_stops = len(distance_matrix_vrp.columns) - 1

        # Extract distances from Depot row
        depot_distances = distance_matrix_vrp.loc['Depot'].copy().drop('Depot')  # Drop self-distance

        # Calculate minimum and maximum distances from the depot
        min_distance = depot_distances.min()
        max_distance = depot_distances.max()

        # Solve VRP problem
        model_collab, current_names_collab = vrp_solver.build_model(
            input_df=input_df_for_vrp,
            chosen_company=chosen_company,
            chosen_candidate=chosen_candidate,
            distance_matrix=distance_matrix_vrp,
            truck_capacity=truck_cap
        )
        solution_collab, routes_collab = vrp_solver.solve(
            m=model_collab,
            max_runtime=1,
            display=False,
            current_names=current_names_collab
        )

        total_distance_collab, avg_distance_per_order_collab = vrp_solver.calculate_distance_per_order(
            routes=routes_collab,
            distance_matrix=distance_matrix_vrp
        )

        # Append results to list, including chosen company and candidate
        results.append({
            "totalsum": total_route_distance,
            "number_stops": num_stops,
            "Max_to_depot": max_distance,
            "Min_to_depot": min_distance,
            "vehicle_cap": truck_cap,
            "greedy_total_sum": total_greedy_dist,
            "real_km_order": avg_distance_per_order_collab,
            "chosen_company": chosen_company,
            "chosen_candidate": chosen_candidate
        })

    # Convert results list to DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV file
    results_df.to_csv("train_df_greedy_osrm.csv", index=False)

    print("Results saved successfully!")