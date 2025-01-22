#########################################################
#                                                       #
# Created on: 15/01/2025                                #
# Created by: Lukas and Wester                          #
#                                                       #
#                                                       #
#########################################################

from Candidate_Ranking.Rankings import CandidateRanking
from Algorithms.distance_calculator import RoadDistanceCalculator
import pandas as pd
import time
import random

TRUCK_CAPACITY = 5
CHOSEN_COMPANY = "Dynamic Industries"
CHOSEN_CANDIDATE = "NextGen Technologies"
LONG_DEPOT = 5.26860985
LAT_DEPOT = 52.2517788

if __name__ == "__main__":
    ###get the data
    input_df = pd.read_csv("../Data/manyLarge.csv")
    input_df_original = input_df.copy()
    input_df['name'] = input_df.groupby('name').cumcount().add(1).astype(str).radd(input_df['name'] + "_")

    # Get Distance matrix
    start_time = time.time()
    calculator = RoadDistanceCalculator()
    distance_matrix = calculator.calculate_distance_matrix(
        input_df, chosen_company=CHOSEN_COMPANY,
        candidate_name=None, method="haversine", computed_distances_df=None)

    square_matrix = calculator.calculate_square_matrix(input_df)
    print('time square matrix', time.time() - start_time)
    algorithms = CandidateRanking()
    start_time = time.time()
    # algorithm1 = algorithms.greedy(distance_matrix, comparing = False)
    algorithm2 = algorithms.bounding_box(input_df_original,distance_matrix,comparing= False)
    #algorithm3 = algorithms.k_means(input_df_original, input_df, distance_matrix, square_matrix, weighted=False)
    # algorithm4 = algorithms.dbscan(input_df_original, distance_matrix)
    # percentages = algorithms.features_dbscan(input_df_original, input_df,distance_matrix, square_matrix)
    print(algorithm2)

    # tune_dbscan = algorithms.dbscan_tuning_silscore(input_df, input_df_original, distance_matrix, square_matrix)

    # features_rbscan = algorithms.features_dbscan(input_df_original, input_df, distance_matrix, square_matrix)

    """
    company_names = input_df['name'].unique()
    algorithms.correlation_df = pd.DataFrame(index=company_names)

    use_method = "bounding box"

    for i in range(5, 13):
        df = pd.read_excel(f"../Data/ranking_results_truck_cap_{i}_haversine_medium.xlsx", sheet_name="Ranks")
        df.set_index('Unnamed: 0', inplace=True)
        algorithms.compare(df,input_df1,i, input_df,method=use_method)
    print(algorithms.correlation_df)
    algorithms.correlation_df.to_excel("../Data/Bounding box approx ranking.xlsx", index=True)
    """

    print(time.time() - start_time)
