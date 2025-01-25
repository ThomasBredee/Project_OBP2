#########################################################
#                                                       #
# Created on: 20/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
# Created on: 25/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
#########################################################

from Algorithms.distance_calculator import RoadDistanceCalculator
from Algorithms.solver_pyvrp import VRPSolver
from Candidate_Ranking.ranking_methods import CandidateRanking
from Input_Transformation.transforming_input import TransformInput
from Expected_gain_models.get_features_training_data import DataFramePreparer
import random
import time
import pandas as pd
from requests.exceptions import ConnectionError
import numpy as np

####### INPUTS FROM THE MODEL VARIABLES
LONG_DEPOT = 5.26860985
LAT_DEPOT = 52.2517788
METHOD = "haversine"
RANKING = "greedy"
AMOUNT_OF_ROWS_TO_GENERATE = 1
MAX_TRUCK_CAP = 20

if __name__ == "__main__":


    ####prepare and merge all data
    # Load the data
    input_df1 = pd.read_csv("Data/mini.csv")
    input_df2 = pd.read_csv("Data/medium.csv")
    input_df3 = pd.read_csv("Data/many.csv")
    input_df4 = pd.read_csv("Data/manyLarge.csv")

    #Combine all dataframes, preserving duplicates within the same DataFrame
    dfs = [input_df1, input_df2,input_df4, input_df3]

    ###make correct split (make sure to save the splitted dfs)
    preparer = DataFramePreparer()
    training_df, testing_df = preparer.clean_and_split_data(dfs)
    training_df.to_csv(f"training_split_{RANKING}_{METHOD}.csv")
    testing_df.to_csv(f"test_split_{RANKING}_{METHOD}.csv")

    #Transform Input to correct format
    check_road_proximity = False #Set true if OSRM container running
    transformer = TransformInput(check_road_proximity=check_road_proximity)
    input_df_modified = transformer.execute_validations(training_df)

    ###create full distance matrix
    distance_calc = RoadDistanceCalculator()
    input_df_modified_with_depot= distance_calc.add_depot(input_df_modified, LAT_DEPOT, LONG_DEPOT)
    full_distance_matrix = distance_calc.calculate_full_distance_matrix(input_df_modified_with_depot, method=METHOD)
    full_distance_matrix_ranking = full_distance_matrix.drop('Depot', axis=1)

    rows = []
    unique_companies = training_df['name'].unique()
    # Add depot to the input dataframe
    for row in range(0, AMOUNT_OF_ROWS_TO_GENERATE):
        print('Making row:', row)

        #Generate random chosen company and candidate
        chosen_company = random.choice(unique_companies)
        chosen_candidate = random.choice([company for company in unique_companies if company != chosen_company])

        row_filter_ranking = full_distance_matrix.index.to_series().apply(
            lambda x: chosen_company in x)
        distance_matrix_ranking = full_distance_matrix.loc[row_filter_ranking, full_distance_matrix_ranking.columns]

        row_filter_vrp = full_distance_matrix.index.to_series().apply(
            lambda x: chosen_company in x or chosen_candidate in x or 'Depot' in x)
        column_filter = full_distance_matrix.columns.to_series().apply(
            lambda x: chosen_company in x or chosen_candidate in x or 'Depot' in x)
        distance_matrix_vrp = full_distance_matrix.loc[row_filter_vrp, column_filter]

        row = preparer.get_features_greedy(input_df_modified, distance_matrix_ranking, distance_matrix_vrp, MAX_TRUCK_CAP, chosen_company, chosen_candidate)
        rows.append(row)

    # Convert results list to DataFrame
    results_df = pd.DataFrame(rows)

    # Save to CSV file
    results_df.to_csv(f"training_data_{RANKING}_{METHOD}.csv", index=False)

    print("Results saved successfully!")