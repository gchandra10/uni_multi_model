import requests, pandas as pd
df = pd.DataFrame({"ENGINESIZE":[2.0,3.0,4.0]})
r = requests.post(
    "http://127.0.0.1:5009/invocations",
    json={"inputs": {"columns": list(df.columns), "data": df.values.tolist()}},
    timeout=10,
)
print(r.status_code, r.text)
