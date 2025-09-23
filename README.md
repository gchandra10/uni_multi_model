# UV - VSCode - Notebook

## Creating a Fresh Project

Install UV

uv init uni_multi_model
cd uni_multi_model

uv add --dev ipykernel

uv run ipython kernel install --user --name=uni_multi_model

uv add matplotlib
uv add mlflow
uv add numpy
uv add pandas
uv add scikit-learn

Open VSCode
Add folder uni_multi_model folder
Save Workspace

create a new .ipynb file

Click Select Kernel (Right Top)

Select Existing Jupyter Server

Click on the Jupyter Kernel with name created above uni_multi_model


