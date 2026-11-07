# Evidence index

* `local/`: actual Windows DVC/MLflow/training/API/test execution. An explicit local assessment
  records Airflow/Kubernetes as unavailable if those services are absent; that is not deployment evidence.
* `live/`: the separate GitHub Actions integration runner's actual Ollama, Airflow 3,
  MLflow, DVC and Kubernetes results. Five conversations are numbered 01–05; failure injections
  and recoveries are under `live/failures/`.
* `live/status.json`: only emitted after all five conversations obtain their required successful
  tool responses. Inspect the associated workflow run for independent execution logs.

No synthetic response is presented as a real run. Mocked unit-test data lives in `tests/`.
Evidence is exported JSON rather than screenshots so exact hashes, run IDs, states, metrics
and diagnostic messages are reviewable and machine-checkable. Runtime credentials, kubeconfig,
databases and model binaries are ignored and must never be committed.

