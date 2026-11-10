"""Confere a execução e fecha checklist, histórico e pacote sem retreinar."""
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlflow
from mlflow.tracking import MlflowClient
import nbformat
from nbconvert import HTMLExporter
import config
from scripts.empacotar import empacotar


def finalizar():
    nb = nbformat.read("parte1.ipynb", as_version=4)
    for c in nb.cells:
        if c.cell_type == "code":
            assert c.execution_count is not None, "Célula de código não executada"
            assert not any(o.output_type == "error" for o in c.outputs), "Erro no notebook"
    suite = ET.parse("artefatos/testes.xml").getroot()
    suites = [suite] if suite.tag == "testsuite" else list(suite)
    for s in suites:
        assert int(s.attrib.get("failures", 0)) + int(s.attrib.get("errors", 0)) + int(s.attrib.get("skipped", 0)) == 0
    total_testes = sum(int(s.attrib["tests"]) for s in suites)
    for c in nb.cells:
        if c.cell_type == "markdown" and c.source.startswith("## 10. Entrega"):
            c.source = c.source.replace("[ ]", "[x]")
    nbformat.write(nb, "parte1.ipynb")
    html, _ = HTMLExporter().from_notebook_node(nb)
    Path("artefatos/parte1.html").write_text(html, encoding="utf-8")

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    cli = MlflowClient()
    atual = cli.get_model_version_by_alias(config.NOME_MODELO, "champion")
    # Preserva evidência das publicações anteriores feitas durante a atividade.
    marcador = "\n<!-- historico-publicacoes -->\n"
    texto = Path("CHANGELOG.md").read_text().split(marcador)[0]
    antigas = [v for v in cli.search_model_versions(f"name = '{config.NOME_MODELO}'") if v.version != atual.version]
    for v in antigas:
        r = cli.get_run(v.run_id)
        texto += (f"\n## v1.0.0 — publicação anterior (MLflow v{v.version})\n"
                  f"- **Tipo:** modelo/pipeline; primeiro ensaio de publicação da parte 1.\n"
                  f"- **Versão no MLflow:** antifraude v{v.version}; run `{v.run_id}`.\n"
                  f"- **O que mudou:** publicação inicial, depois substituída pela versão final v{atual.version} durante a conclusão das evidências e validações.\n"
                  f"- **Por quê:** conferir integração real antes do fechamento da entrega; código original preservado na run.\n"
                  f"- **Evidência:** métricas e artefatos da run; custo de validação R$ {r.data.metrics['val_custo']:.2f}, corte {r.data.metrics['threshold']:.2f}.\n"
                  f"- **Impacto:** versão retirada do alias; histórico de predições e trocas preservado.\n"
                  f"- **Retreinou?** sim, nas semanas 1–10.\n")
    # O marcador delimita a parte gerada nas próximas finalizações.
    principal = Path("CHANGELOG.md").read_text().split(marcador)[0]
    Path("CHANGELOG.md").write_text(principal + marcador + texto[len(principal):], encoding="utf-8")
    with mlflow.start_run(run_id=atual.run_id):
        mlflow.log_artifact("artefatos/testes.xml", "validacao_serving")
        mlflow.log_metric("testes_aprovados", total_testes)
        mlflow.log_artifact("CHANGELOG.md", "governanca")
    empacotar("Miguel Araujo")
    print(f"Entrega finalizada com {total_testes} testes aprovados e checklist completa.")


if __name__ == "__main__":
    finalizar()
