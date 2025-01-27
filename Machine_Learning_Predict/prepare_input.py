#########################################################
#                                                       #
# Created on: 22/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
#########################################################

from VRP_Solver.distance_calculator import RoadDistanceCalculator
from VRP_Solver.solver_pyvrp import VRPSolver
from Candidate_Ranking.ranking_methods import CandidateRanking
import pandas as pd


class PrepareInput:
    def __init__(self):
        pass

    def prep_greedy(self, input_df_with_depot, chosen_company, chosen_candidates, method, distance_matrix_ranking,
                    truck_capacity, greedy_ranking):
        """
        Prepare input data for greedy algorithm by calculating distances and stops.

        Args:
            input_df_with_depot (pd.DataFrame): DataFrame including the depot.
            chosen_company (str): The selected company.
            chosen_candidates (list): List of candidate locations.
            distance_matrix_ranking (pd.DataFrame): Precomputed distance matrix.
            truck_capacity (int): Capacity of the vehicle.
            greedy_ranking (pd.DataFrame): Greedy ranking of candidates.

        Returns:
            pd.DataFrame: Processed results with distance metrics and ranking information.
        """

        results = []

        for chosen_candidate in chosen_candidates:
            distance_calc = RoadDistanceCalculator()
            total_greedy_dist = greedy_ranking.loc[chosen_candidate][0]
            distance_matrix_vrp = distance_calc.calculate_distance_matrix(
                input_df_with_depot,
                chosen_company=chosen_company,
                candidate_name=chosen_candidate,
                method=method,
                computed_distances_df=distance_matrix_ranking
            )
            # Create route sequence based on the filtered column order
            route_sequence = distance_matrix_vrp.index.tolist()

            # Calculate total route distance by summing sequential movements
            total_route_distance = sum(
                distance_matrix_vrp.loc[route_sequence[i], route_sequence[i + 1]]
                for i in range(len(route_sequence) - 1)
            )

            # Calculate the number of stops (excluding the depot itself)
            num_stops = len(distance_matrix_vrp.columns) - 1

            # Extract distances from Depot row
            depot_distances = distance_matrix_vrp.loc['Depot'].drop('Depot')  # Drop self-distance

            # Calculate minimum and maximum distances from the depot
            min_distance = depot_distances.min()
            max_distance = depot_distances.max()

            results.append({
                "chosen_company": chosen_company,
                "chosen_candidate": chosen_candidate,
                "totalsum": total_route_distance,
                "number_stops": num_stops,
                "Max_to_depot": max_distance,
                "Min_to_depot": min_distance,
                "vehicle_cap": truck_capacity,
                "greedy_total_sum": total_greedy_dist
            })

        # Convert results list to DataFrame
        results_df = pd.DataFrame(results)

        return results_df
