# UV - VSCode - Notebook

## Fork and Clone this repo

```
cd uni_multi_model
uv sync
```

**Start the Mlflow Server**

```
mlflow server --host 127.0.0.1 --port 8080 \
--backend-store-uri sqlite:///mlflow.db 
```

**Run the Model**

```
uv run python 01-lr-model.py
```

**Compare sklearn vs pyfunc**

```
uv run python 03_load_test_model.py
```

## Model Serving

**Option 1: Using mlflow models serve**

```

export MLFLOW_TRACKING_URI=http://127.0.0.1:8080

mlflow models serve \
  -m "models:/Linear_Regression_Model/1" \
  --host 127.0.0.1 \
  --port 5001 \
  --env-manager local
```

```

curl -X POST "http://127.0.0.1:5001/invocations" \
  -H "Content-Type: application/json" \
  --data '{"inputs": [{"ENGINESIZE": 2.0}, {"ENGINESIZE": 3.0}, {"ENGINESIZE": 4.0}]}'

OR

curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{
        "dataframe_split": {
          "columns": ["ENGINESIZE"],
          "data": [[2.0],[3.0],[4.0]]
        }
      }'

```
----

**Option 2 : Using Fast API**

```
export MLFLOW_TRACKING_URI=http://127.0.0.1:8080
```

```
uvicorn fast_app:app --host 127.0.0.1 --port 5002
```

```
curl -X POST "http://127.0.0.1:5002/predict-co2-emission" \
  -H "Content-Type: application/json" \
  --data '{"inputs": [{"ENGINESIZE": 2.0}, {"ENGINESIZE": 3.0}, {"ENGINESIZE": 4.0}]}'
```

----

## How to create a NEW UV Project with .py

```
uv init uni_multi_model
cd uni_multi_model

uv add matplotlib mlflow numpy pandas scikit-learn 'uvicorn[standard]' fastapi pydantic
```

- Open VSCode
- Add folder uni_multi_model folder
- Save Workspace
- Create a new .py file and continue your work


## How to create a New UV Project with Notebooks

**Install UV**

```
uv init uni_multi_model
cd uni_multi_model

uv add --dev ipykernel

uv run ipython kernel install --user --name=uni_multi_model

uv add matplotlib mlflow numpy pandas scikit-learn 'uvicorn[standard]' fastapi pydantic
```
- Open VSCode
- Add folder uni_multi_model folder
- Save Workspace

**- Close VSCode and Reopen the Saved Workspace**

- Create a new .ipynb file
- Click Select Kernel (Right Top)
- Select Existing Jupyter Server
- Click on the Jupyter Kernel with name created above uni_multi_model
