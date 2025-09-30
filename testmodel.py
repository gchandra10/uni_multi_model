
import pandas as pd
import mlflow
import mlflow.sklearn


mlflow.set_tracking_uri("http://127.0.0.1:8080")

print("----- UNIVARIATE MODEL -----")
loaded = mlflow.sklearn.load_model("models:/Linear_Regression_Model/1")
X = pd.DataFrame({"ENGINESIZE": [3.5]})
print(loaded.predict(X))

print("----- MULTIVARIATE MODEL -----")
# Load the registered model (version 1 in this case)
loaded = mlflow.sklearn.load_model("models:/MultivariateLinearRegressionModel/1")

# Build a DataFrame with ALL required features
X = pd.DataFrame([{
    "ENGINESIZE": 3.5,
    "CYLINDERS": 6,
    "FUELCONSUMPTION_CITY": 12.5,
    "FUELCONSUMPTION_HWY": 8.7,
    "FUELCONSUMPTION_COMB": 10.6,
    "FUELCONSUMPTION_COMB_MPG": 27,
    "MAKE": "FORD",
    "MODEL": "F150",
    "VEHICLECLASS": "Pickup",
    "TRANSMISSION": "AS6",
    "FUELTYPE": "Z"    # e.g. Z = premium gas in your dataset
}])

print(loaded.predict(X))

