"""Exportação local auditável; não altera o MLflow nem os dados do sistema."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import sys
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlflow
from mlflow.tracking import MlflowClient
import requests
import config


def empacotar(nome="Miguel Araujo"):
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    cli = MlflowClient()
    mv = cli.get_model_version_by_alias(config.NOME_MODELO, "champion")
    run = cli.get_run(mv.run_id)
    saude = requests.get(f"{config.SISTEMA_URL}/saude", timeout=60)
    saude.raise_for_status()
    saude = saude.json()
    avisos = []
    for tag in ["threshold", "motivo"]:
        if tag not in mv.tags:
            avisos.append(f"Tag de versão ausente: {tag}")
    if not mv.description:
        avisos.append("Descrição da versão ausente")
    ref = cli.list_artifacts(mv.run_id, "referencia")
    if not ref:
        avisos.append("Referência ausente")
    if not saude["modelo"].get("ok") or str(saude["modelo"].get("versao")) != mv.version:
        avisos.append("Sistema fora da versão champion")
    if "## v1.0.0" not in Path("CHANGELOG.md").read_text():
        avisos.append("Changelog ausente")
    assert not avisos, avisos
    pasta = Path("entrega")
    if pasta.exists():
        shutil.rmtree(pasta)
    pasta.mkdir()
    evidencias = {"aluno": nome, "gerado_em": datetime.datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat(),
                  "modelo": config.NOME_MODELO, "alias": "champion", "versao": mv.version,
                  "descricao": mv.description, "tags_versao": dict(mv.tags), "run_id": mv.run_id,
                  "experimento": run.info.experiment_id, "params": run.data.params,
                  "metricas": run.data.metrics, "referencia": [x.path for x in ref],
                  "sistema": saude, "avisos": avisos}
    (pasta / "evidencias_mlflow.json").write_text(json.dumps(evidencias, indent=2, ensure_ascii=False))
    runs = cli.search_runs([run.info.experiment_id])
    historico_runs = [{"run_id": r.info.run_id, "status": r.info.status, "params": r.data.params,
                      "metricas": r.data.metrics, "tags": r.data.tags} for r in runs]
    (pasta / "runs_mlflow.json").write_text(json.dumps(historico_runs, indent=2, ensure_ascii=False))
    versoes = [{"versao": v.version, "run_id": v.run_id, "descricao": v.description, "tags": dict(v.tags)}
               for v in cli.search_model_versions(f"name = '{config.NOME_MODELO}'")]
    (pasta / "registry_mlflow.json").write_text(json.dumps(versoes, indent=2, ensure_ascii=False))
    mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{config.NOME_MODELO}@champion", dst_path=str(pasta / "modelo_exportado"))
    mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="referencia", dst_path=str(pasta))
    for p in ["parte1.ipynb", "features.py", "monitoramento.py", "workflow.py", "config.py", "CHANGELOG.md",
              "README.md", "requirements.txt", "requirements-dev.txt", "requirements-lock.txt", "compose.yaml", "Dockerfile", "Makefile", "pytest.ini",
              ".env.exemplo", ".dockerignore", "demanda_parte1.md"]:
        if Path(p).exists():
            shutil.copy2(p, pasta / p)
    for p in ["contrato", "sql", "scripts", "tests", "app", "docker", "artefatos"]:
        if Path(p).exists():
            shutil.copytree(p, pasta / p, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    # CSVs de histórico são necessários para reproduzir treino e notebook.
    shutil.copytree("dados", pasta / "dados")
    for endpoint, nome_csv in [("painel.csv", "painel.csv"), ("decisoes/exportar.csv", "decisoes.csv")]:
        r = requests.get(f"{config.SISTEMA_URL}/api/{endpoint}", timeout=60)
        r.raise_for_status()
        (pasta / nome_csv).write_text(r.text, encoding="utf-8")
    manifesto = {str(p.relative_to(pasta)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(pasta.rglob("*")) if p.is_file()}
    (pasta / "manifesto_sha256.json").write_text(json.dumps(manifesto, indent=2))
    nome_zip = "entrega_parte1_" + "_".join(nome.lower().split())
    shutil.make_archive(nome_zip, "zip", pasta)
    print(f"Champion antifraude v{mv.version}; threshold {mv.tags['threshold']}; nenhum aviso.")
    print(f"Entregue {nome_zip}.zip ({len(manifesto)} arquivos auditáveis).")


if __name__ == "__main__":
    empacotar()
