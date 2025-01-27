#########################################################
#                                                       #
# Created on: 25/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
#########################################################

import pandas as pd
import numpy as np
from Candidate_Ranking.ranking_methods import CandidateRanking
from VRP_Solver.solver_pyvrp import VRPSolver
import random
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler


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

    def get_features_bounding_circle(self, df_input, df_input_modified,  sliced_distance_matrix_ranking, sliced_distance_matrix_vrp, max_truck_cap, chosen_company, chosen_candidate):

        #Get random truck capacity
        truck_cap = random.choice(list(range(2, max_truck_cap+1, 1)))

        #Get total greedy distance
        ranking = CandidateRanking()
        # Get correct format for matrix
        cleaned_matrix = ranking.clean_column_and_index_labels(sliced_distance_matrix_ranking)

        # Get correct dfs
        partner_names = cleaned_matrix.columns.unique().drop(chosen_company)
        company_df = df_input[df_input['name'] == chosen_company]
        partner_df = df_input[df_input['name'] != chosen_company]

        # Get middel point circle
        middel_point_circle = (company_df['lat'].mean(), company_df['lon'].mean())
        distances = company_df.apply(
            lambda row: ranking._euclidean_distance(middel_point_circle[0], middel_point_circle[1], row['lat'],
                                                    row['lon']), axis=1)
        radius = distances.mean()

        # Creating ranking df
        ranking_df = pd.DataFrame({'Percentage Overlap': 0}, index=partner_names)
        partner_names = partner_names.drop(partner_names[partner_names == 'Depot'])
        # Calculate overlap percentages
        for partner in partner_names:
            df_temp = partner_df[partner_df['name'] == partner]

            count = 0
            for _, row in df_temp.iterrows():
                distance = ranking._euclidean_distance(middel_point_circle[0], middel_point_circle[1], row['lat'],
                                                       row['lon'])
                if distance <= radius:
                    count += 1
            ranking_df.loc[partner, 'Percentage Overlap'] = count / len(df_temp)

        ranking = ranking_df.sort_values(by=['Percentage Overlap'], ascending=False)

        # Calculate features
        average_deliveries_in_circle = ranking[['Percentage Overlap']].mean()
        mean_distance_within_circle = distances[distances <= radius].mean()
        area_of_circle = np.pi * (radius ** 2)
        delivery_density = len(distances[distances <= radius]) / area_of_circle
        mean_distance_to_midpoint = distances.mean()


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
            input_df=df_input_modified,
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
            "average_percentage_in_circle": average_deliveries_in_circle[0],
            "mean_distance_within_circle": mean_distance_within_circle,
            "delivery_density": delivery_density,
            "radius": mean_distance_to_midpoint,
            "real_km_order": avg_distance_per_order_collab,
            "chosen_company": chosen_company,
            "chosen_candidate": chosen_candidate
        }

        return results_row


    def get_features_k_means(self, df_input, df_input_modified , full_squared_matrix,
                                     sliced_distance_matrix_vrp, max_truck_cap, chosen_company, chosen_candidate):

        truck_cap = random.choice(list(range(2, max_truck_cap + 1, 1)))
        # Get correct format for matrix
        ranking = CandidateRanking()
        cleaned_matrix = ranking.clean_column_and_index_labels(full_squared_matrix)

        # Prep calculation
        minimal_clusters = len(df_input[df_input['name'] == chosen_company])
        partner_names = cleaned_matrix.columns.unique().drop(chosen_company)
        scaler = StandardScaler()
        df_input[['lat_scaled', 'lon_scaled']] = scaler.fit_transform(df_input[['lat', 'lon']])

        # Determine number of clusters
        silhouette_scores_kmeans = {}
        b_s = {}
        a_s = {}
        for i in range(minimal_clusters, minimal_clusters + 6):
            kmeans = KMeans(n_clusters=i, random_state=42)
            df_new = df_input_modified.copy()
            df_new['cluster'] = kmeans.fit_predict(df_input[['lat_scaled', 'lon_scaled']])
            b_s[i] = ranking._average_distance_between_clusters(df_new, full_squared_matrix)
            silhouette_scores_kmeans[i], a_s[i] = ranking._mean_intra_cluster_distance(df_new, full_squared_matrix, b_s[i],i)

        # Determine optimal clusters
        optimal_clusters = max(silhouette_scores_kmeans, key=silhouette_scores_kmeans.get) + minimal_clusters
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        df_input['cluster'] = kmeans.fit_predict(df_input[['lat_scaled', 'lon_scaled']])
        clusters_company = df_input[df_input['name'] == chosen_company]['cluster'].unique()

        a_max = max(silhouette_scores_kmeans, key=silhouette_scores_kmeans.get)
        a = a_s[a_max]
        b = b_s[a_max]

        # Creating ranking df
        ranking_df = pd.DataFrame({'Percentage Overlap': 0}, index=partner_names)

        # Get percentage overlap of clusters
        for partner in partner_names:
            partner_df = df_input[df_input['name'] == partner]
            partner_data = partner_df[partner_df['name'] == partner]
            is_in_cluster = partner_data['cluster'].isin(clusters_company)

            # Calculate and assign percentage overlap directly to ranking_df
            ranking_df.loc[partner, 'Percentage Overlap'] = is_in_cluster.mean()
        average_overlap = ranking_df[['Percentage Overlap']].mean()

        # Compute average cluster size
        average_cluster_size = df_input['cluster'].value_counts().mean()

        # Compute company centroid distance
        centroid_distances = []
        for cluster in df_input['cluster'].unique():
            cluster_points = df_input[df_input['cluster'] == cluster][['lat_scaled', 'lon_scaled']]
            centroid = kmeans.cluster_centers_[cluster]
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            centroid_distances.extend(distances)
        average_centroid_distance = np.mean(centroid_distances)

        # Compute average minimal and maximal distances within clusters
        min_distances = []
        max_distances = []
        for cluster in df_input['cluster'].unique():
            cluster_points = df_input[df_input['cluster'] == cluster][['lat_scaled', 'lon_scaled']]

            # Only calculate distances if the cluster has more than one point
            if len(cluster_points) > 1:
                pairwise_dist = pdist(cluster_points)
                min_distances.append(np.min(pairwise_dist))
                max_distances.append(np.max(pairwise_dist))

        # Avoid computing mean of empty lists
        average_minimal_distance = np.mean(min_distances) if min_distances else 0
        average_maximal_distance = np.mean(max_distances) if max_distances else 0

        # Get total route lenght based on distance matrix by summing sequential movements
        route_sequence = sliced_distance_matrix_vrp.index.tolist()
        total_route_distance = sum(
            sliced_distance_matrix_vrp.loc[route_sequence[i], route_sequence[i + 1]]
            for i in range(len(route_sequence) - 1)
        )

        # Get the number of locations
        num_stops = len(sliced_distance_matrix_vrp.columns) - 1

        # Calculate minimum and maximum distances from the depot
        depot_distances = sliced_distance_matrix_vrp.loc['Depot'].copy().drop('Depot')  # Drop self-distance
        min_distance = depot_distances.min()
        max_distance = depot_distances.max()

        # Solve the vrp to get real distance per order
        vrp_solver = VRPSolver()
        model_collab, current_names_collab = vrp_solver.build_model(
            input_df=df_input_modified,
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

        # Make row
        results_row = {
            "totalsum": total_route_distance,
            "number_stops": num_stops,
            "Max_to_depot": max_distance,
            "Min_to_depot": min_distance,
            "average_overlap_percentage": average_overlap[0],
            "average_cluster_size": average_cluster_size,
            "average_centroid_distance": average_centroid_distance,
            "average_distance_within_clusters": a,
            "average_distance_between_clusters": b,
            "average_minimal_distance": average_minimal_distance,
            "average_maximal_distance": average_maximal_distance,
            "vehicle_cap": truck_cap,
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