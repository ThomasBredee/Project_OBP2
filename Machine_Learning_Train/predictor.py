#########################################################
#                                                       #
# Created on: 21/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
#########################################################
import pandas as pd
import joblib

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