# Engineering defense / evaluation guide

1. **Why an LLM?** Natural-language tool selection uses Ollama's supported structured tool API.
   The orchestration and permissions remain ordinary tested Python. The LLM cannot fabricate
   output fields or bypass schemas, confirmations or namespace checks.
2. **Why DVC plus Git?** Git versions code and small metadata; DVC content-addresses dataset bytes.
   MLflow stores both the Git reference and exact content hash, allowing independent verification.
3. **Why MLflowClient?** It is the supported Python interface; no UI scraping or guessed run values.
   Ranking considers finished runs with finite values and a declared metric direction/window.
4. **Why Airflow 3?** The TaskFlow DAG models retries and dependencies explicitly. The client uses
   the version 2 public REST paths, bearer authentication and required trigger logical date.
5. **Why readiness and liveness separately?** A responsive process can still lack a usable model.
   A missing model returns readiness 503 and stays out of ready Service endpoints; restarting a
   healthy process alone cannot fix an incorrect model path.
6. **Why no automatic retraining?** Failed deployments do not prove model degradation. Retraining
   recommendations require a recorded failed quality gate, explicit age policy or degradation tag.
   Assessments never call a mutation tool.
7. **How do you know what is deployed?** The approved JSON model has an MLflow run ID and DVC
   provenance. The image is stamped into the Deployment; `/version` returns the loaded model's
   identity. A missing/mismatching deployment identity is reported as uncertain.
8. **How are failures handled?** Timeouts, invalid requests, permission errors and service failures
   are explicit evidence statuses. Read-only evidence collection continues across unavailable
   systems. A response with no successful tools has no operational facts.
9. **What does confirmation protect?** A proposal is stored server-side with exact typed arguments,
   TTL and a one-use ID. The authenticated user must submit the exact confirmation; LLM-generated
   `confirmed=true`, another namespace or another deployment is rejected before any SDK mutation.
10. **Unexpected failure example:** DVC rejects another `.dvc` file as a pipeline dependency.
    The original validation-stage declaration was corrected to depend on the actual dataset
    content and training code; validation itself independently verifies committed metadata.
    This was found by executing real `dvc add`, not by relying only on unit tests.

Evaluation sequence: inspect dataset versions → compare MLflow runs → inspect a real successful
Airflow run and its five tasks → inspect Kubernetes health and failure logs → review the
four-system assessment. Show both injected failures and their recovery records. For a genuinely
new failure during evaluation, collect fresh status and logs; do not assume the injected examples
explain every outage.
