<<<<<<< HEAD
#########################################################
#                                                       #
# Created on: 20/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
# Created on: 26/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
#########################################################

#Imports
from VRP_Solver.distance_calculator import RoadDistanceCalculator
from Input_Transformation.transforming_input import TransformInput
from Machine_Learning_Train.get_features_training_data import DataFramePreparer
import random
import pandas as pd


####### INPUTS FROM THE MODEL VARIABLES
LONG_DEPOT = 5.26860985
LAT_DEPOT = 52.2517788
METHOD = "osrm"
RANKING = "dbscan" #bounding_circle or #greedy
AMOUNT_OF_ROWS_TO_GENERATE = 10_000
MAX_TRUCK_CAP = 20
EPS = 15
MIN_SAMPLES = 2

if __name__ == "__main__":

    ####prepare and merge all data
    # Load the data
    input_df1 = pd.read_csv("Data/mini.csv")
    input_df2 = pd.read_csv("Data/medium.csv")
    input_df3 = pd.read_csv("Data/many.csv")
    input_df4 = pd.read_csv("Data/manyLarge.csv")

    #Combine all dataframes, preserving duplicates within the same DataFrame
    dfs = [input_df1, input_df2, input_df4, input_df3]



    ###make correct split (make sure to save the split dfs)
    preparer = DataFramePreparer()
    training_df, testing_df = preparer.clean_and_split_data(dfs)
    training_df.to_csv(f"training_split_{RANKING}_{METHOD}.csv")
    testing_df.to_csv(f"test_split_{RANKING}_{METHOD}.csv")

    #Transform Input to correct format
    check_road_proximity = False #Set true if OSRM container running
    transformer = TransformInput(check_road_proximity=check_road_proximity)
    input_df_modified = transformer.execute_validations(training_df)
    training_df = transformer.drop_duplicates(training_df)

    ###create full distance matrix
    distance_calc = RoadDistanceCalculator()
    input_df_modified_with_depot= distance_calc.add_depot(input_df_modified, LAT_DEPOT, LONG_DEPOT)
    full_distance_matrix = distance_calc.calculate_full_distance_matrix(input_df_modified_with_depot, method=METHOD)
    full_distance_matrix_ranking = full_distance_matrix.drop('Depot', axis=1)

    if RANKING == "greedy":
        full_distance_matrix = distance_calc.calculate_full_distance_matrix(input_df_modified_with_depot, method=METHOD)
        full_distance_matrix_ranking = full_distance_matrix.drop('Depot', axis=1)
    elif RANKING == "bounding_circle":
        full_distance_matrix = distance_calc.calculate_full_distance_matrix(input_df_modified_with_depot, method=METHOD)
        full_distance_matrix_ranking = full_distance_matrix.drop('Depot', axis=1)
    elif RANKING == "k_means":
        df_input_clustering = transformer.drop_duplicates(training_df)
        full_distance_matrix = distance_calc.calculate_full_distance_matrix(input_df_modified_with_depot, method=METHOD)
        squared_distance_df_kmeans = distance_calc.calculate_square_matrix(input_df_modified)
    elif RANKING == "dbscan":
        df_input_clustering = transformer.drop_duplicates(training_df)
        full_distance_matrix = distance_calc.calculate_full_distance_matrix(input_df_modified_with_depot, method=METHOD)
        squared_distance_df_dbscan = distance_calc.calculate_square_matrix(input_df_modified)

    rows = []
    unique_companies = training_df['name'].unique()
    for row in range(0, AMOUNT_OF_ROWS_TO_GENERATE):
        print('Making row:', row)

        #Generate random chosen company and candidate
        chosen_company = random.choice(unique_companies)
        chosen_candidate = random.choice([company for company in unique_companies if company != chosen_company])

        row_filter_vrp = full_distance_matrix.index.to_series().apply(lambda x: chosen_company in x or chosen_candidate in x or 'Depot' in x)
        column_filter = full_distance_matrix.columns.to_series().apply(lambda x: chosen_company in x or chosen_candidate in x or 'Depot' in x)
        distance_matrix_vrp = full_distance_matrix.loc[row_filter_vrp, column_filter]

        if RANKING == "greedy":
            row_filter_ranking = full_distance_matrix.index.to_series().apply(lambda x: chosen_company in x)
            distance_matrix_ranking = full_distance_matrix.loc[row_filter_ranking, full_distance_matrix_ranking.columns]
            row = preparer.get_features_greedy(input_df_modified, distance_matrix_ranking, distance_matrix_vrp, MAX_TRUCK_CAP, chosen_company, chosen_candidate)

        elif RANKING == "bounding_circle":
            row_filter_ranking = full_distance_matrix.index.to_series().apply(lambda x: chosen_company in x)
            distance_matrix_ranking = full_distance_matrix.loc[row_filter_ranking, full_distance_matrix_ranking.columns]
            row = preparer.get_features_bounding_circle(training_df, input_df_modified, distance_matrix_ranking, distance_matrix_vrp, MAX_TRUCK_CAP,chosen_company, chosen_candidate)

        elif RANKING == "k_means":
            df_to_train = preparer.get_split_training_data(training_df, chosen_company, chosen_candidate, 0.4, 1)
            df_to_train_modified = df_to_train.copy()
            df_to_train_modified = transformer._add_underscore(df_to_train_modified)
            row = preparer.get_features_k_means(df_to_train, df_to_train_modified, squared_distance_df_kmeans, distance_matrix_vrp, MAX_TRUCK_CAP, chosen_company, chosen_candidate)

        elif RANKING == "dbscan":
            df_to_train = preparer.get_split_training_data(training_df, chosen_company, chosen_candidate, 0.4, 1)
            df_to_train_modified = df_to_train.copy()
            df_to_train_modified = transformer._add_underscore(df_to_train_modified)
            df_to_train_modified = transformer.drop_duplicates(df_to_train_modified)
            squared_distance_df_dbscan_copy = squared_distance_df_dbscan.copy()
            squared_distance_df_dbscan_copy = transformer._slice_df(df_to_train_modified,
                                                                    squared_distance_df_dbscan_copy)

            row = preparer.get_features_dbscan(df_to_train, df_to_train_modified, squared_distance_df_dbscan_copy,
                                               distance_matrix_vrp, MAX_TRUCK_CAP, chosen_company, chosen_candidate,
                                               EPS, MIN_SAMPLES)

        rows.append(row)

    # Convert results list to DataFrame
    results_df = pd.DataFrame(rows)

    # Save to CSV file
    results_df.to_csv(f"generated_training_data_{RANKING}_{METHOD}_{AMOUNT_OF_ROWS_TO_GENERATE}_rows.csv", index=False)

=======
#########################################################
#                                                       #
# Created on: 20/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
# Created on: 26/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
#########################################################

#Imports
from VRP_Solver.distance_calculator import RoadDistanceCalculator
from Input_Transformation.transforming_input import TransformInput
from Machine_Learning_Train.get_features_training_data import DataFramePreparer
import random
import pandas as pd


####### INPUTS FROM THE MODEL VARIABLES
LONG_DEPOT = 5.26860985
LAT_DEPOT = 52.2517788
METHOD = "osrm"
RANKING = "dbscan" #bounding_circle or #greedy
AMOUNT_OF_ROWS_TO_GENERATE = 10_000
MAX_TRUCK_CAP = 20
EPS = 15
MIN_SAMPLES = 2

if __name__ == "__main__":

    ####prepare and merge all data
    # Load the data
    input_df1 = pd.read_csv("Data/mini.csv")
    input_df2 = pd.read_csv("Data/medium.csv")
    input_df3 = pd.read_csv("Data/many.csv")
    input_df4 = pd.read_csv("Data/manyLarge.csv")

    #Combine all dataframes, preserving duplicates within the same DataFrame
    dfs = [input_df1, input_df2, input_df4, input_df3]



    ###make correct split (make sure to save the split dfs)
    preparer = DataFramePreparer()
    training_df, testing_df = preparer.clean_and_split_data(dfs)
    training_df.to_csv(f"training_split_{RANKING}_{METHOD}.csv")
    testing_df.to_csv(f"test_split_{RANKING}_{METHOD}.csv")

    #Transform Input to correct format
    check_road_proximity = False #Set true if OSRM container running
    transformer = TransformInput(check_road_proximity=check_road_proximity)
    input_df_modified = transformer.execute_validations(training_df)
    training_df = transformer.drop_duplicates(training_df)

    ###create full distance matrix
    distance_calc = RoadDistanceCalculator()
    input_df_modified_with_depot= distance_calc.add_depot(input_df_modified, LAT_DEPOT, LONG_DEPOT)
    full_distance_matrix = distance_calc.calculate_full_distance_matrix(input_df_modified_with_depot, method=METHOD)
    full_distance_matrix_ranking = full_distance_matrix.drop('Depot', axis=1)

    if RANKING == "greedy":
        full_distance_matrix = distance_calc.calculate_full_distance_matrix(input_df_modified_with_depot, method=METHOD)
        full_distance_matrix_ranking = full_distance_matrix.drop('Depot', axis=1)
    elif RANKING == "bounding_circle":
        full_distance_matrix = distance_calc.calculate_full_distance_matrix(input_df_modified_with_depot, method=METHOD)
        full_distance_matrix_ranking = full_distance_matrix.drop('Depot', axis=1)
    elif RANKING == "k_means":
        df_input_clustering = transformer.drop_duplicates(training_df)
        full_distance_matrix = distance_calc.calculate_full_distance_matrix(input_df_modified_with_depot, method=METHOD)
        squared_distance_df_kmeans = distance_calc.calculate_square_matrix(input_df_modified)
    elif RANKING == "dbscan":
        df_input_clustering = transformer.drop_duplicates(training_df)
        full_distance_matrix = distance_calc.calculate_full_distance_matrix(input_df_modified_with_depot, method=METHOD)
        squared_distance_df_dbscan = distance_calc.calculate_square_matrix(input_df_modified)

    rows = []
    unique_companies = training_df['name'].unique()
    for row in range(0, AMOUNT_OF_ROWS_TO_GENERATE):
        print('Making row:', row)

        #Generate random chosen company and candidate
        chosen_company = random.choice(unique_companies)
        chosen_candidate = random.choice([company for company in unique_companies if company != chosen_company])

        row_filter_vrp = full_distance_matrix.index.to_series().apply(lambda x: chosen_company in x or chosen_candidate in x or 'Depot' in x)
        column_filter = full_distance_matrix.columns.to_series().apply(lambda x: chosen_company in x or chosen_candidate in x or 'Depot' in x)
        distance_matrix_vrp = full_distance_matrix.loc[row_filter_vrp, column_filter]

        if RANKING == "greedy":
            row_filter_ranking = full_distance_matrix.index.to_series().apply(lambda x: chosen_company in x)
            distance_matrix_ranking = full_distance_matrix.loc[row_filter_ranking, full_distance_matrix_ranking.columns]
            row = preparer.get_features_greedy(input_df_modified, distance_matrix_ranking, distance_matrix_vrp, MAX_TRUCK_CAP, chosen_company, chosen_candidate)

        elif RANKING == "bounding_circle":
            row_filter_ranking = full_distance_matrix.index.to_series().apply(lambda x: chosen_company in x)
            distance_matrix_ranking = full_distance_matrix.loc[row_filter_ranking, full_distance_matrix_ranking.columns]
            row = preparer.get_features_bounding_circle(training_df, input_df_modified, distance_matrix_ranking, distance_matrix_vrp, MAX_TRUCK_CAP,chosen_company, chosen_candidate)

        elif RANKING == "k_means":
            df_to_train = preparer.get_split_training_data(training_df, chosen_company, chosen_candidate, 0.4, 1)
            df_to_train_modified = df_to_train.copy()
            df_to_train_modified = transformer._add_underscore(df_to_train_modified)
            row = preparer.get_features_k_means(df_to_train, df_to_train_modified, squared_distance_df_kmeans, distance_matrix_vrp, MAX_TRUCK_CAP, chosen_company, chosen_candidate)

        elif RANKING == "dbscan":
            df_to_train = preparer.get_split_training_data(training_df, chosen_company, chosen_candidate, 0.4, 1)
            df_to_train_modified = df_to_train.copy()
            df_to_train_modified = transformer._add_underscore(df_to_train_modified)
            df_to_train_modified = transformer.drop_duplicates(df_to_train_modified)
            squared_distance_df_dbscan_copy = squared_distance_df_dbscan.copy()
            squared_distance_df_dbscan_copy = transformer._slice_df(df_to_train_modified,
                                                                    squared_distance_df_dbscan_copy)

            row = preparer.get_features_dbscan(df_to_train, df_to_train_modified, squared_distance_df_dbscan_copy,
                                               distance_matrix_vrp, MAX_TRUCK_CAP, chosen_company, chosen_candidate,
                                               EPS, MIN_SAMPLES)

        rows.append(row)

    # Convert results list to DataFrame
    results_df = pd.DataFrame(rows)

    # Save to CSV file
    results_df.to_csv(f"generated_training_data_{RANKING}_{METHOD}_{AMOUNT_OF_ROWS_TO_GENERATE}_rows.csv", index=False)

>>>>>>> 995f455a1d37e0b2f24f53d70f716ae0df4c79b8
    print("Results saved successfully!")