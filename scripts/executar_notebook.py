"""Executa, salva o notebook completo e reempacota incluindo todas as saídas."""
from pathlib import Path
import sys
import subprocess
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

raiz = Path(__file__).resolve().parents[1]
path = raiz / "parte1.ipynb"
nb = nbformat.read(path, as_version=4)
for c in nb.cells:
    if c.cell_type == "code":
        c.outputs = []
        c.execution_count = None

def progresso(cell, cell_index, **kwargs):
    print(f"Executando célula {cell_index}: {cell.source[:90].splitlines()[0]}", flush=True)

client = NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(raiz)}},
                        on_cell_start=progresso)
try:
    client.execute()
finally:
    for c in nb.cells:
        resposta = c.metadata.get("resposta")
        arquivo = {"escolha": "justificativa.txt", "limites": "limites_justificativa.txt"}.get(resposta)
        if arquivo and (raiz / "artefatos" / arquivo).exists():
            c.source = (raiz / "artefatos" / arquivo).read_text(encoding="utf-8")
    nbformat.write(nb, path)
html, _ = HTMLExporter().from_notebook_node(nb)
(raiz / "artefatos" / "parte1.html").write_text(html, encoding="utf-8")
subprocess.run([sys.executable, "-m", "pytest", "-q", "--junitxml=artefatos/testes.xml"], check=True)
from scripts.finalizar_entrega import finalizar
finalizar()
print("Notebook executado e salvo; testes, HTML, checklist e ZIP atualizados.")
