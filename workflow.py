"""Operações rastreáveis da parte 1; a narrativa e escolhas ficam no notebook."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import subprocess

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import requests

import config
from app import contrato as contrato_app
from features import colunas_contrato, validar_features, criar_pipeline
from monitoramento import metricas, custo_decisoes, varrer_thresholds

ARTEFATOS = Path("artefatos")
EXPERIMENTO = "Aurora | Parte 1 | custo e contrato v1.0"
SEED = 42


def salvar_json(caminho, dados):
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    Path(caminho).write_text(json.dumps(dados, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def carregar_historico(contrato):
    tx = pd.read_csv("dados/transacoes_semanas_01_12.csv", dtype={"id_transacao": str, "id_cliente": str})
    rot = pd.read_csv("dados/rotulos_semanas_01_12.csv", dtype={"id_transacao": str})
    assert tx.id_transacao.is_unique and rot.id_transacao.is_unique, "IDs duplicados"
    assert set(tx.id_transacao) == set(rot.id_transacao), "IDs dos rótulos divergem"
    assert set(tx.semana) == set(range(1, 13))
    auditoria = []
    for semana, lote in tx.groupby("semana"):
        _, erros = contrato_app.validar(lote, contrato)
        validar_features(lote, contrato, estrito=False)
        auditoria.append({"semana": int(semana), "linhas": len(lote), "erros": " | ".join(erros), "ok": not erros})
    assert all(x["ok"] for x in auditoria), auditoria
    historico = tx.merge(rot, on="id_transacao", how="inner", validate="one_to_one")
    assert len(historico) == len(tx) and historico.fraude.isin([0, 1]).all()
    # A razão foi entregue arredondada a quatro casas: não a recalculamos.
    erro_razao = (historico.razao_valor_media - historico.valor_transacao / historico.valor_medio_30d).abs()
    assert erro_razao.max() <= 0.00005001, erro_razao.max()
    datas = pd.to_datetime(historico.data_transacao)
    assert (datas.dt.hour == historico.hora_transacao).all()
    assert (datas.dt.dayofweek == historico.dia_semana).all()
    ARTEFATOS.mkdir(exist_ok=True)
    auditoria = pd.DataFrame(auditoria)
    auditoria.to_csv(ARTEFATOS / "auditoria_contrato.csv", index=False)
    return historico, auditoria


def linhagem(contrato):
    arquivos = ["dados/transacoes_semanas_01_12.csv", "dados/rotulos_semanas_01_12.csv",
                "contrato/contrato_features.yaml", "features.py", "monitoramento.py", "workflow.py",
                "requirements.txt", "requirements-dev.txt", "scripts/preparar_notebook.py"]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = "git indisponível no container; hashes de código são a fonte de proveniência"
    return {"seed": SEED, "contrato": contrato["versao"], "commit_base": commit,
            "sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in arquivos},
            "pacotes": {p: importlib.metadata.version(p) for p in
                        ["mlflow", "scikit-learn", "numpy", "pandas", "scipy", "sqlalchemy"]}}


def treinar_candidato(nome, estimador, treino, validacao, contrato):
    nomes, _, _ = colunas_contrato(contrato)
    modelo = criar_pipeline(estimador, contrato)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENTO)
    with mlflow.start_run(run_name=nome) as run:
        mlflow.log_params({"algoritmo": type(estimador).__name__, "treino_semanas": "1-10",
                           "validacao_semanas": "11-12", "contrato_versao": contrato["versao"],
                           "features_n": len(nomes), "seed": SEED,
                           "selecao": "menor custo total de validacao; threshold grade 0.01-0.99",
                           **{f"estimador_{k}": v for k, v in estimador.get_params().items()}})
        mlflow.set_tags({"versao_semantica": "1.0.0", "projeto": "Banco Aurora",
                         "avaliacao": "validacao usada na selecao; nao e teste independente"})
        modelo.fit(treino[nomes], treino.fraude)
        scores = modelo.predict_proba(validacao[nomes])[:, 1]
        curva = varrer_thresholds(validacao, scores)
        melhor = curva.iloc[0].to_dict()
        padrao = metricas(validacao.fraude, scores, validacao.valor_transacao, 0.5)
        mlflow.log_metrics({"threshold": melhor["threshold"], **{f"val_{k}": float(v) for k, v in melhor.items() if k != "threshold"},
                            **{f"padrao_05_{k}": float(v) for k, v in padrao.items()}})
        exemplo = treino[nomes].head(5)
        # Assinatura descreve a saída sklearn predict; a API usa predict_proba.
        mlflow.sklearn.log_model(
            modelo, "modelo", signature=infer_signature(exemplo, modelo.predict(exemplo)),
            input_example=exemplo, code_paths=["features.py"],
            pip_requirements=[f"{p}=={importlib.metadata.version(p)}" for p in
                              ["mlflow", "scikit-learn", "numpy", "pandas", "scipy", "pyyaml", "cloudpickle"]],
        )
        mlflow.log_dict(linhagem(contrato), "linhagem.json")
        mlflow.log_dict(contrato, "contrato.json")
        mlflow.log_artifact("contrato/contrato_features.yaml", "codigo")
        for p in ["features.py", "monitoramento.py", "workflow.py", "requirements.txt", "scripts/preparar_notebook.py"]:
            mlflow.log_artifact(p, "codigo")
        pasta = ARTEFATOS / nome
        pasta.mkdir(exist_ok=True)
        curva.to_csv(pasta / "thresholds.csv", index=False)
        mlflow.log_artifact(str(pasta / "thresholds.csv"))
        print(f"{nome}: PR-AUC={melhor['pr_auc']:.4f}; threshold={melhor['threshold']:.2f}; "
              f"custo R$ {melhor['custo']:,.2f} / 2 semanas; run={run.info.run_id}")
        return {"nome": nome, "modelo": modelo, "scores": scores, "curva": curva,
                "run_id": run.info.run_id, "metricas": melhor, "padrao": padrao}


def comparar(candidatos, validacao):
    y, valores = validacao.fraude, validacao.valor_transacao
    regra = (validacao.canal == "pix") & (valores > 1000)
    linhas = [{"alternativa": "Aprovar tudo", "threshold": np.nan, "custo": custo_decisoes(y, np.zeros(len(y)), valores)},
              {"alternativa": "Regra Pix > R$ 1.000", "threshold": np.nan, "custo": custo_decisoes(y, regra, valores)}]
    for c in candidatos:
        linhas.extend([
            {"alternativa": c["nome"] + " | custo mínimo", **c["metricas"]},
            {"alternativa": c["nome"] + " | padrão 0,5", "threshold": 0.5, **c["padrao"]},
        ])
    tabela = pd.DataFrame(linhas)
    tabela["custo_medio_semanal"] = tabela.custo / validacao.semana.nunique()
    tabela.to_csv(ARTEFATOS / "comparacao.csv", index=False)
    return tabela


def registrar_campeao(campeao, justificativa, contrato):
    """Reexecução da mesma run reutiliza a versão; nova run cria nova versão."""
    cli = MlflowClient()
    versoes = cli.search_model_versions(f"name = '{config.NOME_MODELO}'")
    mv = next((v for v in versoes if v.run_id == campeao["run_id"]), None)
    if mv is None:
        mv = mlflow.register_model(f"runs:/{campeao['run_id']}/modelo", config.NOME_MODELO)
    t = float(campeao["metricas"]["threshold"])
    for k, v in {"threshold": t, "motivo": "v1.0.0 - menor custo de validação entre os candidatos",
                 "contrato_versao": contrato["versao"], "versao_semantica": "1.0.0"}.items():
        cli.set_model_version_tag(config.NOME_MODELO, mv.version, k, str(v))
    cli.update_model_version(config.NOME_MODELO, mv.version, description=justificativa)
    cli.set_registered_model_alias(config.NOME_MODELO, config.ALIAS_PRODUCAO, mv.version)
    cli.set_tag(campeao["run_id"], "selecionado", "champion")
    return cli.get_model_version(config.NOME_MODELO, mv.version)


def validar_serving(validacao, contrato):
    nomes, _, _ = colunas_contrato(contrato)
    x = validacao[validacao.semana == 12][nomes].head(5)
    modelo = mlflow.sklearn.load_model(f"models:/{config.NOME_MODELO}@champion")
    scores = modelo.predict_proba(x)[:, 1]
    r = requests.post(f"{config.SISTEMA_URL}/predict", json=x.to_dict("records"), timeout=60)
    r.raise_for_status()
    remoto = pd.DataFrame(r.json())
    # A API fornecida arredonda a seis casas. A igualdade aqui respeita essa
    # serialização, e o teste de lotes compara a precisão completa no banco.
    np.testing.assert_array_equal(remoto.score.to_numpy(), np.array([round(float(s), 6) for s in scores]))
    np.testing.assert_array_equal(remoto.bloquear, scores >= remoto.threshold.to_numpy())
    remoto["score_mlflow_integral"] = scores
    remoto["score_mlflow_6_casas"] = [round(float(s), 6) for s in scores]
    remoto.to_csv(ARTEFATOS / "paridade_predict.csv", index=False)
    return remoto


def carregar_validacao_sistema(validacao):
    """Carrega somente semanas de validação, como demonstração histórica."""
    r = requests.get(f"{config.SISTEMA_URL}/api/lotes", timeout=30)
    r.raise_for_status()
    existentes = {x["semana"] for x in r.json()}
    registros = []
    cols = contrato_app.carregar()["metadados"] + contrato_app.nomes_features()
    for semana, g in validacao.groupby("semana"):
        if int(semana) not in existentes:
            r = requests.post(f"{config.SISTEMA_URL}/api/lotes", files={
                "arquivo": (f"historico_semana_{semana}.csv", g[cols].to_csv(index=False).encode(), "text/csv")}, timeout=120)
        else:
            r = requests.post(f"{config.SISTEMA_URL}/api/lotes/{semana}/repontuar", timeout=120)
        r.raise_for_status()
        registros.append(r.json())
    r = requests.post(f"{config.SISTEMA_URL}/api/rotulos", files={"arquivo": (
        "rotulos_historicos_11_12.csv", validacao[["id_transacao", "fraude", "data_contestacao"]].to_csv(index=False).encode(), "text/csv")}, timeout=120)
    r.raise_for_status()
    salvar_json(ARTEFATOS / "ingestao_validacao.json", {"lotes": registros, "rotulos": r.json()})
    return registros


def verificar_lote_integral(validacao, campeao, contrato):
    import io
    nomes = colunas_contrato(contrato)[0]
    evidencias = []
    for semana, g in validacao.groupby("semana"):
        r = requests.get(f"{config.SISTEMA_URL}/api/semanas/{semana}/dados.csv", timeout=120)
        r.raise_for_status()
        remoto = pd.read_csv(io.StringIO(r.text))
        ordenado = g.set_index("id_transacao").loc[remoto.id_transacao]
        esperado = campeao["modelo"].predict_proba(ordenado[nomes])[:, 1]
        np.testing.assert_allclose(remoto.score, esperado, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(remoto.bloqueada, esperado >= campeao["metricas"]["threshold"])
        evidencias.append({"semana": int(semana), "n": len(remoto), "erro_maximo": float(np.max(np.abs(remoto.score - esperado))),
                           "decisoes_identicas": True})
    salvar_json(ARTEFATOS / "paridade_lotes.json", evidencias)
    return pd.DataFrame(evidencias)
