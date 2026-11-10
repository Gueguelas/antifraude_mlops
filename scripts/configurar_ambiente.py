"""Cria segredos locais somente na primeira execução, sem imprimi-los."""
from pathlib import Path
import secrets

p = Path(__file__).resolve().parents[1] / ".env"
if not p.exists():
    p.write_text(
        f"PG_SENHA={secrets.token_hex(16)}\n"
        f"JUPYTER_TOKEN={secrets.token_hex(24)}\n"
        "PG_HOST=127.0.0.1\nPG_PORT=54321\nPG_USER=postgres\n"
        "PG_DB_APP=antifraude\nPG_DB_MLFLOW=mlflow\n"
        "MLFLOW_TRACKING_URI=http://127.0.0.1:5000\n"
        "SISTEMA_URL=http://127.0.0.1:8000\nNOME_MODELO=antifraude\n",
        encoding="utf-8",
    )
    p.chmod(0o600)
    print(".env local criado (permissão 600).")
else:
    print(".env existente preservado.")
