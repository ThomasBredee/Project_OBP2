#########################################################
#                                                       #
# Created on: 21/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
#########################################################

import pandas as pd
import joblib  # For saving and loading models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


class ModelTrainer:
    def __init__(self, model_type="linear_regression"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        X = df[['totalsum', 'number_stops', 'Max_to_depot', 'Min_to_depot', 'vehicle_cap', 'greedy_total_sum']]
        y = df['real_km_order']
        return X, y

    def train(self, file_path):
        X, y = self.load_data(file_path)

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Choose the model
        if self.model_type == "linear_regression":
            self.model = LinearRegression()
        elif self.model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError("Unsupported model type! Choose 'linear_regression' or 'random_forest'.")

        # Train the model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate the model
        train_predictions = self.model.predict(X_train_scaled)
        test_predictions = self.model.predict(X_test_scaled)

        train_mse = mean_squared_error(y_train, train_predictions)
        test_mse = mean_squared_error(y_test, test_predictions)

        print(f"Model: {self.model_type}")
        print(f"Train MSE: {train_mse:.2f}")
        print(f"Test MSE: {test_mse:.2f}")

        # Save the trained model and scaler
        joblib.dump(self.model, f"{self.model_type}_model_greedy_osrm.pkl")
        joblib.dump(self.scaler, "scaler_greedy_osrm.pkl")

        return self.model, self.scaler


# Example usage
if __name__ == "__main__":
    trainer = ModelTrainer(model_type="random_forest")  # Change to "linear_regression" if needed
    trainer.train("Machine_Learning_Train/osrm/TrainData/train_df_greedy_osrm.csv")
