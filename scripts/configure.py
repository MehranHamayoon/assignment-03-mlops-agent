import os
import secrets
from pathlib import Path


def main():
    path = Path(".env")
    if path.exists():
        print(".env already exists; existing credentials preserved.")
        return
    text = Path(".env.example").read_text(encoding="utf-8")
    uid = os.getuid() if hasattr(os, "getuid") else 50000
    text = text.replace("AIRFLOW_UID=50000\n", f"AIRFLOW_UID={uid}\n")
    text = text.replace("AIRFLOW_PASSWORD=\n", f"AIRFLOW_PASSWORD={secrets.token_urlsafe(24)}\n")
    text = text.replace("POSTGRES_PASSWORD=\n", f"POSTGRES_PASSWORD={secrets.token_hex(24)}\n")
    text = text.replace("AGENT_API_KEY=\n", f"AGENT_API_KEY={secrets.token_urlsafe(32)}\n")
    path.write_text(text, encoding="utf-8")
    print("Created ignored .env with random local credentials (values are not printed).")


if __name__ == "__main__":
    main()
