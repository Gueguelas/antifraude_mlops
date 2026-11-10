import io
import json
import subprocess
import sys
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pytest
import requests
import config
from features import carregar_contrato, colunas_contrato

pytestmark = pytest.mark.integracao


def test_champion_rastreavel_e_servido():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    cli = MlflowClient()
    mv = cli.get_model_version_by_alias(config.NOME_MODELO, "champion")
    assert mv.description and "motivo" in mv.tags
    assert 0.01 <= float(mv.tags["threshold"]) <= 0.99
    assert cli.list_artifacts(mv.run_id, "referencia")
    assert cli.get_run(mv.run_id).data.params["treino_semanas"] == "1-10"
    r = requests.get(f"{config.SISTEMA_URL}/api/modelo", timeout=60)
    r.raise_for_status()
    assert r.json()["modelo"]["versao"] == mv.version
    assert r.json()["modelo"]["run_id"] == mv.run_id


def test_paridade_lote_completo_e_painel():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    modelo = mlflow.sklearn.load_model("models:/antifraude@champion")
    cli = MlflowClient()
    t = float(cli.get_model_version_by_alias("antifraude", "champion").tags["threshold"])
    nomes = colunas_contrato(carregar_contrato())[0]
    r = requests.get(f"{config.SISTEMA_URL}/api/semanas/12/dados.csv", timeout=120)
    r.raise_for_status()
    lote = pd.read_csv(io.StringIO(r.text))
    s = modelo.predict_proba(lote[nomes])[:, 1]
    np.testing.assert_allclose(lote.score, s, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(lote.bloqueada, s >= t)
    # Seleciona casos dos dois extremos do ranking, não apenas legítimas fáceis.
    indices = np.r_[np.argsort(s)[:3], np.argsort(s)[-3:]]
    resp = requests.post(f"{config.SISTEMA_URL}/predict", json=lote.iloc[indices][nomes].to_dict("records"), timeout=60)
    resp.raise_for_status()
    np.testing.assert_array_equal([x["score"] for x in resp.json()], [round(float(x), 6) for x in s[indices]])
    np.testing.assert_array_equal([x["bloquear"] for x in resp.json()], s[indices] >= t)
    painel = requests.get(f"{config.SISTEMA_URL}/api/painel", timeout=60)
    painel.raise_for_status()
    assert {x["semana"] for x in painel.json()} >= {11, 12}


def test_api_rejeita_feature_ausente_e_categoria_invalida():
    nomes = colunas_contrato(carregar_contrato())[0]
    x = pd.read_csv("dados/transacoes_semanas_01_12.csv", nrows=1)[nomes].to_dict("records")[0]
    ruim = dict(x, canal="boleto")
    assert requests.post(f"{config.SISTEMA_URL}/predict", json=[ruim], timeout=60).status_code == 422
    x.pop("canal")
    assert requests.post(f"{config.SISTEMA_URL}/predict", json=[x], timeout=60).status_code == 422


def test_modelo_exportavel_em_processo_isolado_sem_workspace(tmp_path):
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    caminho = mlflow.artifacts.download_artifacts(artifact_uri="models:/antifraude@champion", dst_path=str(tmp_path))
    nomes = colunas_contrato(carregar_contrato())[0]
    x = pd.read_csv("dados/transacoes_semanas_01_12.csv", nrows=5)[nomes]
    entrada = tmp_path / "entrada.json"
    entrada.write_text(x.to_json(orient="records"))
    # -I remove cwd/PYTHONPATH: só o code_paths do próprio artefato pode suprir features.py.
    codigo = """import sys, json, mlflow.sklearn, pandas as pd
modelo = mlflow.sklearn.load_model(sys.argv[1])
x = pd.read_json(sys.argv[2])
print(json.dumps(modelo.predict_proba(x)[:, 1].tolist()))
"""
    processo = subprocess.run([sys.executable, "-I", "-c", codigo, caminho, str(entrada)],
                              cwd=tmp_path, text=True, capture_output=True, check=True, timeout=60)
    esperado = mlflow.sklearn.load_model(caminho).predict_proba(x)[:, 1]
    np.testing.assert_array_equal(json.loads(processo.stdout.strip().splitlines()[-1]), esperado)
