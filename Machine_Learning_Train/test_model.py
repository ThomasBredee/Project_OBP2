#########################################################
#                                                       #
# Created on: 21/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
# Updated on: 29/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
#########################################################

import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from Machine_Learning_Train.create_training_data import MAX_TRUCK_CAP
from Machine_Learning_Train.get_features_training_data import DataFramePreparer
from itertools import permutations
from Input_Transformation.transforming_input import TransformInput
from VRP_Solver.distance_calculator import RoadDistanceCalculator
import random
import pickle

LONG_DEPOT = 5.26860985
LAT_DEPOT = 52.2517788
METHOD = "haversine"
RANKING = "greedy"
MODEL_TYPE = "random_forest"
MAX_TRUCK_CAP = 20
EPS = 15
MIN_SAMPLES = 2


class ModelTester:
    def __init__(self, model_file, scaler_file):
        """
        Initializes the ModelTester with pre-trained model and scaler.
        Args:
        model_file (str): Path to the joblib file containing the trained model.
        scaler_file (str): Path to the joblib file containing the trained scaler.
        """
        self.model = joblib.load(model_file)
        self.scaler = joblib.load(scaler_file)

    def evaluate(self, df, input_features):
        """
        Evaluates the model on the given DataFrame.
        Args:
        df (DataFrame): The DataFrame containing the test data.
        input_features (list of str): Column names in the DataFrame that are features for the model.
        """
        X_test = df[input_features]
        y_test = df['real_km_order']

        # Standardize features using the saved scaler
        X_test_scaled = self.scaler.transform(X_test)

        # Make predictions
        predictions = self.model.predict(X_test_scaled)

        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        divergence = np.abs(y_test - predictions).mean()

        # Print evaluation results
        print("Evaluation Results:")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"R-squared (R²): {r2:.2%}")
        print(f"Average Misprediction (Divergence): {divergence:.2f} km/order")

        # Detailed Predictions vs Actuals
        print("\nDetailed Predictions vs Actuals:")
        for actual, predicted in zip(y_test, predictions):
            print(f"Actual: {actual:.2f}, Predicted: {predicted:.2f}")

    def generate_test_data(self, input_df, input_df_modified, max_truck_cap, method, company_pairs):
        rows = []
        ###create full distance matrix
        distance_calc = RoadDistanceCalculator()
        preparer = DataFramePreparer()
        input_df_modified_with_depot = distance_calc.add_depot(input_df_modified, LAT_DEPOT, LONG_DEPOT)

        if RANKING == "greedy":
            full_distance_matrix = distance_calc.calculate_full_distance_matrix(input_df_modified_with_depot,method=method)
            full_distance_matrix_ranking = full_distance_matrix.drop('Depot', axis=1)
        elif RANKING == "bounding_circle":
            full_distance_matrix = distance_calc.calculate_full_distance_matrix(input_df_modified_with_depot,method=METHOD)
            full_distance_matrix_ranking = full_distance_matrix.drop('Depot', axis=1)
        elif RANKING == "k_means":
            df_input_clustering = transformer.drop_duplicates(input_df)
            full_distance_matrix = distance_calc.calculate_full_distance_matrix(input_df_modified_with_depot,
                                                                                method=METHOD)
            squared_distance_df_kmeans = distance_calc.calculate_square_matrix(input_df_modified)
        elif RANKING == "dbscan":
            df_input_clustering = transformer.drop_duplicates(input_df)
            full_distance_matrix = distance_calc.calculate_full_distance_matrix(input_df_modified_with_depot,
                                                                                method=METHOD)
            squared_distance_df_dbscan = distance_calc.calculate_square_matrix(input_df_modified)

        for pair in company_pairs:
            chosen_company = pair[0]
            chosen_candidate = pair[1]

            row_filter_vrp = full_distance_matrix.index.to_series().apply(
                lambda x: chosen_company in x or chosen_candidate in x or 'Depot' in x)
            column_filter = full_distance_matrix.columns.to_series().apply(
                lambda x: chosen_company in x or chosen_candidate in x or 'Depot' in x)
            distance_matrix_vrp = full_distance_matrix.loc[row_filter_vrp, column_filter]

            if RANKING == "greedy":
                row_filter_ranking = full_distance_matrix.index.to_series().apply(lambda x: chosen_company in x)
                distance_matrix_ranking = full_distance_matrix.loc[
                    row_filter_ranking, full_distance_matrix_ranking.columns]
                row = preparer.get_features_greedy(input_df_modified, distance_matrix_ranking, distance_matrix_vrp,
                                                   max_truck_cap, chosen_company, chosen_candidate)
            elif RANKING == "bounding_circle":
                row_filter_ranking = full_distance_matrix.index.to_series().apply(lambda x: chosen_company in x)
                distance_matrix_ranking = full_distance_matrix.loc[row_filter_ranking, full_distance_matrix_ranking.columns]
                row = preparer.get_features_bounding_circle(input_df, input_df_modified, distance_matrix_ranking,
                                                            distance_matrix_vrp, MAX_TRUCK_CAP, chosen_company,
                                                            chosen_candidate)
            elif RANKING == "k_means":
                row = preparer.get_features_k_means(df_input_clustering, input_df_modified, squared_distance_df_kmeans,
                                                    distance_matrix_vrp, MAX_TRUCK_CAP, chosen_company,
                                                    chosen_candidate)
            elif RANKING == "dbscan":

                row = preparer.get_features_dbscan(df_input_clustering, input_df_modified, squared_distance_df_dbscan,
                                                   distance_matrix_vrp, MAX_TRUCK_CAP, chosen_company, chosen_candidate,
                                                   EPS, MIN_SAMPLES)

            rows.append(row)

         # Convert results list to DataFrame
        results_df = pd.DataFrame(rows)
        return results_df



if __name__ == "__main__":
    path_model = f"Machine_Learning_Train/{METHOD}/TrainedModels/RF/"
    test_model = ModelTester(f"{path_model}{MODEL_TYPE}_model_{METHOD}_{RANKING}.pkl", f"{path_model}scaler_{METHOD}_{RANKING}.pkl")
    test_df = pd.read_csv(f"Machine_Learning_Train/{METHOD}/TestData/test_split_{RANKING}_{METHOD}.csv", index_col=0)
    # Transform Input to correct format
    check_road_proximity = True  # Set true if OSRM container running
    transformer = TransformInput(check_road_proximity=check_road_proximity)
    input_df_modified = transformer.execute_validations(test_df)
    company_pairs = [list(pair) for pair in permutations(test_df['name'].unique(), 2)]
    data_to_predict_on = test_model.generate_test_data(test_df, input_df_modified, MAX_TRUCK_CAP, METHOD, company_pairs)

    data_to_predict_on_cols = data_to_predict_on.drop(columns=["real_km_order", "chosen_company", "chosen_candidate"]).columns

    test_model.evaluate(data_to_predict_on, data_to_predict_on_cols)

    data_to_predict_on.nunique()

    from joblib import load
    path_model = f"Machine_Learning_Train/{METHOD}/TrainedModels/RF/"
    # Load the model from disk using joblib
    model = load(f"{path_model}{MODEL_TYPE}_model_{METHOD}_{RANKING}.pkl")

    # Access and print the feature importances
    feature_importances = model.feature_importances_
    print("Feature Importances:", feature_importances)