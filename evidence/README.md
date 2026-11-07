# Evidence index

The final [hosted execution](https://github.com/MehranHamayoon/assignment-03-mlops-agent/actions/runs/42392077317)
passed all five scenarios and both failure/recovery investigations. Its source commit is
`e3e1339e1d05c301de566fe63a19e20989ef6ea4`. The exported comparison contains exactly five real
runs, selects a recorded F1 candidate, and does not infer production readiness from that score.
See the [readable conversations](../docs/demo-conversations.md) and `live/manifest.json`.

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

