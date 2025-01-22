#########################################################
#                                                       #
# Created on: 21/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
#########################################################

import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class ModelTester:
    def __init__(self, model_file, scaler_file):
        self.model = joblib.load(model_file)
        self.scaler = joblib.load(scaler_file)

    def load_test_data(self, file_path):
        df = pd.read_csv(file_path)  # Test only on 10 rows
        X_test = df[['totalsum', 'number_stops', 'Max_to_depot', 'Min_to_depot', 'vehicle_cap', 'greedy_total_sum']]
        y_test = df['real_km_order']
        return X_test, y_test

    def evaluate(self, file_path):
        X_test, y_test = self.load_test_data(file_path)

        # Standardize features using the saved scaler
        X_test_scaled = self.scaler.transform(X_test)

        # Make predictions
        predictions = self.model.predict(X_test_scaled)

        # Evaluate performance
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # Calculate average misprediction (divergence)
        divergence = np.abs(y_test - predictions).mean()

        print("Evaluation Results:")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"R-squared (R²): {r2:.2%}")
        print(f"Average Misprediction (divergence): {divergence:.2f} km/order")

        # Overfitting Test: Check for large difference between training and test MSE
        if mse > 2 * r2:
            print("Warning: Possible overfitting detected!")

        # Robustness Test: Evaluate stability across test cases
        print("\nDetailed Predictions vs Actuals:")
        for i in range(len(y_test)):
            print(f"Actual: {y_test.iloc[i]:.2f}, Predicted: {predictions[i]:.2f}")

# Example usage
if __name__ == "__main__":
    path = "Expected_gain_models/osrm/TrainedModels/RF/"
    tester = ModelTester(f"{path}random_forest_model_greedy_osrm.pkl", f"{path}scaler_greedy_osrm.pkl")
    tester.evaluate("Expected_gain_models/osrm/TrainData/test_df_greedy_osrm.csv")
