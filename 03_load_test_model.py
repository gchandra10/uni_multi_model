
import time
import pandas as pd
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:8080")

X = pd.DataFrame({"ENGINESIZE": [4.5]})
MODEL_URI = "models:/Linear_Regression_Model/1"

print("----- UNIVARIATE MODEL USING sklearn flavor -----")
start_load = time.time()
sk_model = mlflow.sklearn.load_model(MODEL_URI)
end_load = time.time()

start_pred = time.time()
sk_pred = sk_model.predict(X)
end_pred = time.time()

print(f"Prediction: {sk_pred}")
print(f"Load time: {end_load - start_load:.4f} sec")
print(f"Predict time: {end_pred - start_pred:.6f} sec\n")


print("----- UNIVARIATE MODEL USING pyfunc flavor -----")
start_load1 = time.time()
pyfunc_model = mlflow.pyfunc.load_model(MODEL_URI)
end_load1 = time.time()

start_pred1 = time.time()
pyfunc_pred = pyfunc_model.predict(X)
end_pred1 = time.time()

print(f"Prediction: {pyfunc_pred}")
print(f"Load time: {end_load1 - start_load1:.4f} sec")
print(f"Predict time: {end_pred1 - start_pred1:.6f} sec\n")

print("""
      pyfunc loads faster because it’s lazy — it doesn’t unpack the whole model until it’s actually used.
      
      sklearn loads slower because it eagerly unpacks and validates everything for you.
      """)

# print("----- MULTIVARIATE MODEL -----")
# # Load the registered model (version 1 in this case)
# loaded = mlflow.sklearn.load_model("models:/MultivariateLinearRegressionModel/1")

# # Build a DataFrame with ALL required features
# X = pd.DataFrame([{
#     "ENGINESIZE": 3.5,
#     "CYLINDERS": 6,
#     "FUELCONSUMPTION_CITY": 12.5,
#     "FUELCONSUMPTION_HWY": 8.7,
#     "FUELCONSUMPTION_COMB": 10.6,
#     "FUELCONSUMPTION_COMB_MPG": 27,
#     "MAKE": "FORD",
#     "MODEL": "F150",
#     "VEHICLECLASS": "Pickup",
#     "TRANSMISSION": "AS6",
#     "FUELTYPE": "Z"    # e.g. Z = premium gas in your dataset
# }])

# print(loaded.predict(X))

