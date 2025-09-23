
# mlflow models serve -m runs:/9fd4cc293dcf42be8cbbccd6e6181471/lr_model --port 5001 --env-manager local

import requests
rows = [{"ENGINESIZE": 2.0}, {"ENGINESIZE": 3.0}, {"ENGINESIZE": 4.0}]
r = requests.post(
    "http://127.0.0.1:5001/invocations",
    json={"inputs": rows},
    timeout=10,
)
print(r.status_code, r.text)



# curl -X POST "http://127.0.0.1:5009/invocations" \
#   -H "Content-Type: application/json" \
#   --data '{"inputs": [{"ENGINESIZE": 2.0}, {"ENGINESIZE": 3.0}, {"ENGINESIZE": 4.0}]}'
