
import pandas as pd
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:8080")

loaded = mlflow.sklearn.load_model("models:/Linear_Regression_Model/1")
X = pd.DataFrame({"ENGINESIZE": [3.5]})
print(loaded.predict(X))


# cols = ["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_CITY",
#         "FUELCONSUMPTION_HWY","FUELCONSUMPTION_COMB","FUELCONSUMPTION_COMB_MPG"]
# X = pd.DataFrame([[3.5, 6, 12.1, 8.4, 10.4, 27.0]], columns=cols)
# print(loaded.predict(X))
