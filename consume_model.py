
## Consume the model via REST API

import requests
rows = [{"ENGINESIZE": 2.0}, {"ENGINESIZE": 3.0}, {"ENGINESIZE": 4.0}]
r = requests.post(
    "http://127.0.0.1:5001/invocations",
    json={"inputs": rows},
    timeout=10,
)
print(r.status_code, r.text)
