#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import json, os
from pathlib import Path
Path('/opt/airflow/passwords.json').write_text(json.dumps({'agent': os.environ['AIRFLOW_PASSWORD']}))
PY
git config --global --add safe.directory /opt/project
exec airflow standalone

