#########################################################
#                                                       #
# Created on: 21/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
# Updated on: 27/01/2025                                #
# Created by: Dennis Botman                             #
#                                                       #
#########################################################

import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np

METHOD = "haversine"
RANKING = "dbscan"  #k_means, #bounding_circle
MODEL_TYPE = "random_forest"

class ModelTrainer:
    def __init__(self, model_type="linear_regression"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()

    def train(self, df, train_columns):
        X = df[train_columns]
        y = df['real_km_order']

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Choose the model
        if self.model_type == "linear_regression":
            self.model = LinearRegression()
        elif self.model_type == "random_forest":
            self.model = RandomForestRegressor()
        else:
            raise ValueError("Unsupported model type! Choose 'linear_regression' or 'random_forest'.")

        # Cross-validation setup
        kf = KFold(n_splits=5, shuffle=True)

        # Optimize parameters
        best_params = self.optimize_params(X_scaled, y, kf)
        self.model.set_params(**best_params)

        # Train the model with the best parameters
        self.model.fit(X_scaled, y)

        # Evaluate the model
        predictions = self.model.predict(X_scaled)
        mse = mean_squared_error(y, predictions)
        divergence_per_order = np.sum(np.abs(predictions - y)) / len(y)
        r2 = r2_score(y, predictions)

        print(f"Model: {self.model_type} trained with optimized parameters.")
        print(f"MSE: {mse:.4f}")
        print(f"Divergence per Order: {divergence_per_order:.4f}")
        print(f"R^2 Score: {r2:.4f}")

        joblib.dump(self.model, f"{self.model_type}_model_{METHOD}_{RANKING}.pkl")
        joblib.dump(self.scaler, f"scaler_{METHOD}_{RANKING}.pkl")

        return self.model, self.scaler

    def optimize_params(self, X, y, kf):
        if self.model_type == "random_forest":
            param_dist = {
                'n_estimators': [100, 200, 300, 400, 500],
                'max_depth': [None, 10, 20, 30, 40, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            }
            model = RandomForestRegressor()
        else:
            return {}  # Return an empty dict if not random forest

        # Setup the randomized search with cross-validation
        random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist,
                                           n_iter=100, cv=kf, verbose=1, n_jobs=-1)
        random_search.fit(X, y)
        print(f"Optimized parameters found: {random_search.best_params_}")
        return random_search.best_params_



# Example usage
if __name__ == "__main__":
    trainer = ModelTrainer(model_type=MODEL_TYPE)  # Change to "linear_regression" if needed
    train_data = pd.read_csv(f"Machine_Learning_Train/{METHOD}/TrainData/generated_training_data_{RANKING}_{METHOD}_10000_rows.csv")
    train_df = train_data.drop_duplicates(keep='first')
    training_columns = train_df.drop(columns=["real_km_order", "chosen_company", "chosen_candidate"]).columns
    trainer.train(train_df, training_columns)

    print(training_columns)