
import pandas as pd
import mlflow
loaded = mlflow.sklearn.load_model("runs:/GET_RUNID_FROM_UI/model")
X = pd.DataFrame({"ENGINESIZE": [3.5]})
print(loaded.predict(X))


# cols = ["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_CITY",
#         "FUELCONSUMPTION_HWY","FUELCONSUMPTION_COMB","FUELCONSUMPTION_COMB_MPG"]
# X = pd.DataFrame([[3.5, 6, 12.1, 8.4, 10.4, 27.0]], columns=cols)
# print(loaded.predict(X))
