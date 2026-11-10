"""Lê lote da API e referência da run champion; salva tabelas e alertas locais."""
import argparse
from pathlib import Path
import io
import json
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import requests
import config
import monitoramento as mon
from features import carregar_contrato


def executar(semana, fechados=False):
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    cli = MlflowClient()
    mv = cli.get_model_version_by_alias(config.NOME_MODELO, "champion")
    with tempfile.TemporaryDirectory() as tmp:
        path = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="referencia", dst_path=tmp)
        referencia = pd.read_csv(Path(path) / "referencia.csv")
        meta = json.loads((Path(path) / "metadados.json").read_text())
    contrato = carregar_contrato()
    if meta["run_id"] != mv.run_id or meta["contrato_versao"] != contrato["versao"]:
        raise ValueError("Referência incompatível com run ou contrato vigente.")
    t = float(mv.tags["threshold"])
    if t != meta["threshold"]:
        raise ValueError("Threshold da referência difere da versão champion.")
    if meta["limites_alerta"] != mon.LIMITES:
        raise ValueError("Limites mudaram desde a aprovação. Versione a decisão e a referência antes de prosseguir.")
    r = requests.get(f"{config.SISTEMA_URL}/api/semanas/{semana}/dados.csv", timeout=120)
    r.raise_for_status()
    lote = pd.read_csv(io.StringIO(r.text))
    if lote.empty:
        raise ValueError("Lote vazio ou semana ainda não recebida.")
    if lote.id_transacao.duplicated().any() or not (lote.semana == semana).all():
        raise ValueError("Duplicidade de IDs ou semana inconsistente no lote.")
    if "versao_modelo" not in lote or not (lote.versao_modelo.astype(str) == str(mv.version)).all():
        raise ValueError("Lote sem score champion vigente. Avalie Repontuar e preserve o histórico.")
    if not (lote.threshold == t).all():
        raise ValueError("Threshold das predições não corresponde ao champion.")
    tabelas = mon.relatorio(referencia, lote, contrato, t, rotulos_fechados=fechados)
    destino = Path("artefatos") / "monitoramento" / f"semana_{semana:02d}"
    destino.mkdir(parents=True, exist_ok=True)
    for nome, tabela in tabelas.items():
        tabela.to_csv(destino / f"{nome}.csv", index=False)
        print(f"\n{nome.upper()}\n{tabela.to_string(index=False)}")
    estado = {"semana": semana, "run_id": mv.run_id, "versao": mv.version, "threshold": t,
              "rotulos_fechados": fechados, "alertas": len(tabelas["alertas"]),
              "status": "investigar" if len(tabelas["alertas"]) else "sem_alertas"}
    (destino / "estado.json").write_text(json.dumps(estado, indent=2, ensure_ascii=False))
    return estado


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--semana", type=int, required=True)
    p.add_argument("--rotulos-fechados", action="store_true", help="Use só após confirmação do fechamento e recebimento integral.")
    args = p.parse_args()
    try:
        estado = executar(args.semana, args.rotulos_fechados)
    except Exception as e:
        destino = Path("artefatos/monitoramento") / f"semana_{args.semana:02d}"
        destino.mkdir(parents=True, exist_ok=True)
        (destino / "falha.json").write_text(json.dumps({"status": "falha", "erro": str(e)}, ensure_ascii=False, indent=2))
        print(f"Monitoramento falhou: {e}", file=sys.stderr)
        sys.exit(2)
    sys.exit(1 if estado["alertas"] else 0)
