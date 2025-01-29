#########################################################
#                                                       #
# Created on: 21/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
#########################################################

import pandas as pd
import joblib
import numpy as np

class ModelPredictor:
    def __init__(self, model_file, scaler_file):
        self.model = joblib.load(model_file)
        self.scaler = joblib.load(scaler_file)

    def predict(self, input_data):
        # Convert input data to DataFrame for consistency
        input_df = pd.DataFrame([input_data], columns=['totalsum', 'number_stops', 'Max_to_depot', 'Min_to_depot', 'vehicle_cap', 'greedy_total_sum'])

        # Scale the input data using the pre-trained scaler
        input_scaled = self.scaler.transform(input_df)

        # Make prediction
        prediction = self.model.predict(input_scaled)
        return prediction[0]

# Example usage
if __name__ == "__main__":
    predictor = ModelPredictor("random_forest_model_greedy.pkl", "scaler_greedy.pkl")

    # Sample input to predict
    sample_input = {
        'totalsum': 500,
        'number_stops': 7,
        'Max_to_depot': 150,
        'Min_to_depot': 30,
        'vehicle_cap': 8,
        'greedy_total_sum': 450
    }

    predicted_value = predictor.predict(sample_input)
    print(f"Predicted km/order: {predicted_value:.2f}")
