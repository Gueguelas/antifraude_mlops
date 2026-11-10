"""Preenche o roteiro original; execute só para regenerar o notebook sem saídas."""
import json
from pathlib import Path
import textwrap

p = Path("parte1.ipynb")
nb = json.loads(p.read_text())


def codigo(i, s):
    nb["cells"][i]["source"] = textwrap.dedent(s).strip().splitlines(keepends=True)
    nb["cells"][i]["outputs"] = []
    nb["cells"][i]["execution_count"] = None


def texto(i, s):
    nb["cells"][i]["source"] = textwrap.dedent(s).strip().splitlines(keepends=True)


codigo(4, '''
%matplotlib inline
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
import features
import monitoramento as mon
import workflow as wf

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.figsize": (11, 4.5), "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold"})
pd.set_option("display.max_columns", 20)
contrato = features.carregar_contrato()
nomes, categoricas, numericas = features.colunas_contrato(contrato)
assert len(nomes) == 13
display(pd.DataFrame(contrato["features"]).fillna("—"))
print("Categóricas:", categoricas)
print("Numéricas/booleanas:", numericas)
''')
codigo(5, '''
historico, auditoria = wf.carregar_historico(contrato)
display(auditoria)
print(f"Junção 1:1: {len(historico):,} transações; nenhuma linha perdida.")
print("Nulos nas features:", int(historico[nomes].isna().sum().sum()))
print("Todas as semanas passam no contrato, na razão valor/média e na consistência de hora/dia.")
''')
codigo(6, '''
# data_contestacao é ausente nos legítimos, e é PROIBIDA como feature.
assert not set(nomes) & {"fraude", "id_cliente", "semana", "id_transacao", "data_contestacao", "data_transacao"}
display(historico.groupby("canal").agg(n=("fraude", "size"), taxa_fraude=("fraude", "mean"),
                                      valor_total=("valor_transacao", "sum")))
wf.salvar_json(wf.ARTEFATOS / "linhagem.json", wf.linhagem(contrato))
''')
codigo(8, '''
treino = historico[historico.semana.between(1, 10)].copy()
validacao = historico[historico.semana.isin([11, 12])].copy()
assert treino.semana.max() < validacao.semana.min()
assert set(treino.id_transacao).isdisjoint(validacao.id_transacao)
semanal = historico.groupby("semana").agg(n=("fraude", "size"), fraudes=("fraude", "sum"),
                                           taxa_fraude=("fraude", "mean"))
semanal["conjunto"] = np.where(semanal.index <= 10, "treino", "validação")
display(semanal)
print(f"Treino: {len(treino):,}; validação: {len(validacao):,}")
semanal.to_csv(wf.ARTEFATOS / "populacao_semanal.csv")
''')
codigo(9, '''
ax = (100 * semanal.taxa_fraude).plot.bar(color=["#7c9bb5"] * 10 + ["#007f73"] * 2)
ax.set(title="Fraude por semana • divisão temporal preservada", xlabel="Semana", ylabel="Fraude (%)")
plt.tight_layout()
plt.savefig(wf.ARTEFATOS / "fraude_semanal.png", dpi=160)
plt.show()
''')
texto(10, '''
**Por que não dividir aleatoriamente?** O banco usa passado para decidir sobre
transações futuras. Em uma divisão aleatória, transações de semanas mais recentes
ajudariam a prever semanas antigas, mascarando mudanças de comportamento e
produzindo uma avaliação que não representa a operação. O mesmo cliente pode
aparecer em semanas diferentes; isso é permitido, mas seus IDs não entram no modelo.

As semanas 1–10 ajustam inclusive o pré-processamento. As semanas 11–12 servem
exclusivamente para comparar os candidatos e escolher o threshold pedido no roteiro.
Portanto, os custos reportados são **estimativas de validação usadas na seleção**,
com algum otimismo; não constituem teste independente nem garantia para semanas futuras.
O histórico foi fornecido com rótulos fechados. Em uma simulação cronológica real,
é obrigatório esperar uma semana pelo fechamento de cada lote, inclusive para treinar.
''')
codigo(12, '''
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

alternativas = {
    "regressao_logistica": LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                                             solver="lbfgs", random_state=wf.SEED),
    "hist_gradient_boosting": HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=220, max_leaf_nodes=15, min_samples_leaf=40,
        l2_regularization=2.0, early_stopping=False, random_state=wf.SEED),
}
# Hiperparâmetros fixados previamente; sem busca repetida nas semanas 11–12.
# early_stopping=False evita uma divisão aleatória interna.
display(features.criar_pipeline(alternativas["regressao_logistica"], contrato))
''')
codigo(13, '''
display(Markdown("""Todo o tratamento fica no Pipeline registrado: validação das 13
entradas, log1p e escala nas numéricas, one-hot nas categóricas e classificador.
O one-hot conhece **todos os domínios do YAML**, inclusive UFs ausentes no treino;
uma categoria fora do contrato é rejeitada. Não reconstruímos histórico de clientes
nem recalculamos agregados: usamos exatamente as features cruas da Plataforma.

A regressão logística oferece uma referência simples e regularizada; o boosting
captura interações e efeitos não lineares. O balanceamento da regressão altera a
escala dos scores, por isso o corte é selecionado separadamente para cada modelo.
Não interpretamos esses scores como probabilidades calibradas de risco."""))
''')
codigo(15, '''
candidatos = []
for nome, estimador in alternativas.items():
    candidatos.append(wf.treinar_candidato(nome, estimador, treino, validacao, contrato))
''')
codigo(16, '''
resumo_runs = pd.DataFrame([{"algoritmo": c["nome"], "run_id": c["run_id"],
                             "pr_auc": c["metricas"]["pr_auc"],
                             "recall": c["metricas"]["recall"],
                             "precision": c["metricas"]["precision"],
                             "custo": c["metricas"]["custo"],
                             "threshold": c["metricas"]["threshold"]} for c in candidatos])
display(resumo_runs)
resumo_runs.to_csv(wf.ARTEFATOS / "runs_comparadas.csv", index=False)
display(Markdown("Runs disponíveis no [MLflow local](http://127.0.0.1:5000). Compare custo, PR-AUC e threshold no experimento Aurora."))
''')
codigo(17, '''
# O modelo que custa menos é escolhido, sem usar acurácia como critério.
campeao = min(candidatos, key=lambda c: (c["metricas"]["custo"], c["nome"]))
threshold = float(campeao["metricas"]["threshold"])
print("Vencedor:", campeao["nome"], "| threshold:", threshold)
''')
codigo(19, '''
# Custo: valor das fraudes aprovadas + 15 reais por legítima bloqueada.
from monitoramento import custo_decisoes
assert custo_decisoes([1, 0, 1, 0], [0, 1, 1, 0], [200, 80, 900, 20]) == 215.0
for c in candidatos:
    display(c["curva"].head(5)[["threshold", "custo", "recall", "precision", "legitimas_bloqueadas"]])
''')
codigo(20, '''
fig, ax = plt.subplots()
for c in candidatos:
    curva = c["curva"].sort_values("threshold")
    ax.plot(curva.threshold, curva.custo / 2, label=c["nome"], linewidth=2)
    ax.scatter(c["metricas"]["threshold"], c["metricas"]["custo"] / 2, s=55)
ax.set(xlabel="Threshold (score ≥ corte bloqueia)", ylabel="Custo médio semanal (R$)",
       title="Decisão econômica • thresholds de 0,01 a 0,99")
ax.legend()
plt.tight_layout()
plt.savefig(wf.ARTEFATOS / "custo_threshold.png", dpi=160)
plt.show()
with mlflow.start_run(run_id=campeao["run_id"]):
    mlflow.log_artifact(str(wf.ARTEFATOS / "custo_threshold.png"), "avaliacao")
''')
codigo(21, '''
comparacao = wf.comparar(candidatos, validacao)
display(comparacao[["alternativa", "threshold", "custo", "custo_medio_semanal", "recall", "precision"]])
piso_aprovar, piso_regra = comparacao.iloc[0].custo, comparacao.iloc[1].custo
assert campeao["metricas"]["custo"] < min(piso_aprovar, piso_regra, campeao["padrao"]["custo"]), "Piso de negócio não atendido"
por_semana = pd.DataFrame([{"semana": int(s), **mon.metricas(g.fraude,
    campeao["modelo"].predict_proba(g[nomes])[:, 1], g.valor_transacao, threshold),
    "custo_regra": custo_decisoes(g.fraude, (g.canal == "pix") & (g.valor_transacao > 1000), g.valor_transacao),
    "custo_aprovar": float(g.loc[g.fraude == 1, "valor_transacao"].sum()),
    "custo_padrao_05": mon.metricas(g.fraude, campeao["modelo"].predict_proba(g[nomes])[:, 1], g.valor_transacao, 0.5)["custo"]}
    for s, g in validacao.groupby("semana")])
display(por_semana)
por_semana.to_csv(wf.ARTEFATOS / "custos_semanais.csv", index=False)
assert (por_semana.custo < por_semana[["custo_regra", "custo_aprovar", "custo_padrao_05"]].min(axis=1)).all()
m = campeao["metricas"]
def brl(x):
    return "R$ " + f"{x:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
economia = piso_regra - m["custo"]
justificativa = (
    f"Escolhemos {campeao['nome']}, com corte {threshold:.2f}, pelo menor custo observado. "
    f"Nas semanas 11–12, o custo total é {brl(m['custo'])}, média de {brl(m['custo']/2)} por semana. "
    f"A regra Pix acima de R$ 1.000 custa {brl(piso_regra/2)} por semana; a economia média é "
    f"{brl(economia/2)} por semana ({economia/piso_regra:.1%}). Aprovar tudo custa {brl(piso_aprovar/2)} "
    f"por semana; o mesmo modelo com corte 0,5 custa {brl(campeao['padrao']['custo']/2)}. "
    f"No total das duas semanas, {int(m['legitimas_bloqueadas'])} transações legítimas são bloqueadas "
    f"({brl(15*m['legitimas_bloqueadas'])} de atrito), e passam {int(m['fraudes_aprovadas'])} fraudes "
    f"({brl(m['fraude_perdida'])}). Recall {m['recall']:.1%}, precision {m['precision']:.1%}, "
    f"PR-AUC (average precision) {m['pr_auc']:.4f}, bloqueio {m['taxa_bloqueio']:.1%}. "
    "Esses números são de validação usada na seleção, não promessa de economia futura. "
    "Bloquear encaminha à verificação no app; não significa cancelar a transação."
)
(wf.ARTEFATOS / "justificativa.txt").write_text(justificativa, encoding="utf-8")
display(Markdown(justificativa))
''')
texto(23, '''
A escolha econômica, com custos semanais, economia, atrito e perdas residuais,
está quantificada na tabela e na justificativa acima. A execução salva a
justificativa completa nesta célula antes de gerar a entrega final.
''')
nb["cells"][23]["metadata"]["resposta"] = "escolha"
codigo(25, '''
# A referência vai para a run ANTES de expor a versão como champion.
referencia = validacao.copy()
referencia["score"] = campeao["scores"]
referencia["bloqueada"] = referencia.score >= threshold
referencia["threshold"] = threshold
pasta_ref = wf.ARTEFATOS / "referencia"
pasta_ref.mkdir(exist_ok=True)
referencia.to_csv(pasta_ref / "referencia.csv", index=False)
wf.salvar_json(pasta_ref / "metadados.json", {
    "run_id": campeao["run_id"], "contrato_versao": contrato["versao"], "semanas": [11, 12],
    "threshold": threshold, "n": len(referencia), "rotulos_fechados": True,
    "features": nomes, "limites_alerta": mon.LIMITES,
    "origem": "validação histórica, não lote futuro", "linhagem": wf.linhagem(contrato)})
with mlflow.start_run(run_id=campeao["run_id"]):
    mlflow.log_artifacts(str(pasta_ref), artifact_path="referencia")
    mlflow.log_artifact(str(wf.ARTEFATOS / "comparacao.csv"), "avaliacao")
    mlflow.log_artifact(str(wf.ARTEFATOS / "custos_semanais.csv"), "avaliacao")
    mlflow.log_text(justificativa, "justificativa.txt")
mv = wf.registrar_campeao(campeao, justificativa, contrato)
print("Champion:", config.NOME_MODELO, "versão", mv.version, "| run", mv.run_id)
display(dict(mv.tags))
''')
codigo(26, '''
from mlflow.tracking import MlflowClient
cli = MlflowClient()
registrada = cli.get_model_version_by_alias(config.NOME_MODELO, "champion")
assert registrada.run_id == campeao["run_id"]
assert float(registrada.tags["threshold"]) == threshold
assert registrada.description and registrada.tags["motivo"]
assert cli.list_artifacts(registrada.run_id, "referencia")
print("Registro, tags na VERSÃO, descrição, alias e referência conferidos.")
''')
codigo(28, '''
paridade = wf.validar_serving(validacao, contrato)
display(paridade)
print("5/5 scores idênticos após a serialização da API (seis casas); decisões idênticas.")
print("A API original arredonda score. A seguir verificamos também a precisão completa dos lotes.")
''')
codigo(29, '''
# Dados históricos de demonstração: não simulamos a chegada de semanas futuras.
display(wf.carregar_validacao_sistema(validacao))
display(wf.verificar_lote_integral(validacao, campeao, contrato))
import requests
saude = requests.get(f"{config.SISTEMA_URL}/saude", timeout=60).json()
assert saude["modelo"]["ok"] and saude["modelo"]["versao"] == mv.version
display(saude)
wf.salvar_json(wf.ARTEFATOS / "saude.json", saude)
''')
codigo(31, '''
semana11 = referencia[referencia.semana == 11]
semana12 = referencia[referencia.semana == 12]
zero = mon.relatorio(semana11, semana12, contrato, threshold, rotulos_fechados=True)
pasta_zero = wf.ARTEFATOS / "relatorio_zero"
pasta_zero.mkdir(exist_ok=True)
for nome, tabela in zero.items():
    tabela.to_csv(pasta_zero / f"{nome}.csv", index=False)
    print("\\n", nome.upper())
    display(tabela)
''')
codigo(32, '''
# O monitor lê de onde a operação vai ler nas próximas partes: a API do sistema.
import io
r = requests.get(f"{config.SISTEMA_URL}/api/semanas/12/dados.csv", timeout=120)
r.raise_for_status()
lote_sistema = pd.read_csv(io.StringIO(r.text))
operacao = mon.relatorio(referencia, lote_sistema, contrato, threshold, rotulos_fechados=True)
display(operacao["performance"])
sem_rotulos = lote_sistema.drop(columns="fraude")
pendente = mon.relatorio(referencia, sem_rotulos, contrato, threshold, rotulos_fechados=False)
assert (pendente["performance"].status == "aguardando_rotulos").all()
display(pendente["performance"])
''')
codigo(33, '''
drift_zero = zero["drift"]
menor = drift_zero.sort_values("p_valor").iloc[0]
perda11 = mon.metricas(semana11.fraude, semana11.score, semana11.valor_transacao, threshold)
perda12 = mon.metricas(semana12.fraude, semana12.score, semana12.valor_transacao, threshold)
variacao_custo = perda12["custo_por_1000"] / perda11["custo_por_1000"] - 1
limites_texto = (
    f"No relatório zero (semana 11 versus 12), o menor p-valor é {menor.p_valor:.3g}, "
    f"em {menor.feature}, com efeito {menor.efeito:.4f}; "
    f"{int(drift_zero.alerta.sum())} alertas de drift e {len(zero['alertas'])} alertas totais. "
    f"O custo por mil aumentou {variacao_custo:.1%} da semana 11 para 12: "
    f"{brl(perda11['custo_por_1000'])} para {brl(perda12['custo_por_1000'])}. "
    f"As fraudes aprovadas foram {perda11['fraudes_aprovadas']} e {perda12['fraudes_aprovadas']}, "
    f"mas os valores perdidos foram {brl(perda11['fraude_perdida'])} e {brl(perda12['fraude_perdida'])}. "
    "Logo, recall parecido pode conviver com prejuízo maior. Os alertas de custo são sinais "
    "para investigar os valores das fraudes perdidas e os canais; não provam drift nem "
    "justificam retreino imediato. Mantemos os limites sem ajustá-los para esconder os alertas. "
    "Com cerca de 20 mil observações por semana, diferenças pequenas podem produzir "
    "p-valores baixos; isso não mede prejuízo nem prova mudança de conceito. "
    "Também fazemos 14 testes correlacionados, então controlamos multiplicidade com Benjamini-Hochberg. "
    "Um p-valor pequeno sozinho não dispara alerta. Na idade das contas, a separação "
    "se concentra na cauda superior; o máximo passa de 2.276 a 2.283 dias. Isso é compatível "
    "com envelhecimento das contas e distribuição com concentração na cauda, não prova "
    "de mudança do mecanismo de fraude. "
    "Nas contínuas e no score exigimos D do KS > 0,10 e q < 0,01; nas categóricas, "
    "booleanas, hora e dia exigimos distância de variação total > 0,05 e q < 0,01. "
    "A variação total mede a massa de probabilidade redistribuída. Exigimos pelo menos 500 "
    "observações por amostra; volumes menores ficam inconclusivos. Erros do contrato são críticos, "
    "independentemente de teste estatístico. As fronteiras de alerta foram escolhidas "
    "pela relevância operacional e devem ser reavaliadas com mais semanas estáveis. "
    "Na saída, alertamos mudança absoluta de bloqueio > 2 pontos percentuais. "
    "Com rótulos explicitamente fechados e cobertura integral, alertamos custo por mil > 25% "
    "acima da referência ou queda de recall > 10 pontos percentuais, globalmente e por canal, "
    "com ao menos 500 transações e 10 fraudes no lote/segmento. PR-AUC e precision também "
    "são reportadas, sem limiar rígido por ora. Drift gera investigação; não retreino automático. "
    "Sem rótulos, monitoramos qualidade, drift e saída; custo/recall aguardam a semana seguinte. "
    "O dado precisa estar fechado: ausência de contestação parcial não autoriza tratá-lo como legítimo. "
    "Os testes supõem amostras independentes; clientes recorrentes e testes correlacionados "
    "limitam a interpretação formal dos p-valores. O relatório zero usa semanas distintas; "
    "a leitura da semana 12 contra a referência 11–12 é só demonstração operacional, "
    "pois há sobreposição. A detecção operacional válida começa em lotes novos sem essa sobreposição."
)
(wf.ARTEFATOS / "limites_justificativa.txt").write_text(limites_texto, encoding="utf-8")
display(Markdown(limites_texto))
''')
codigo(34, '''
from datetime import datetime
from zoneinfo import ZoneInfo
data = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%d/%m/%Y")
changelog = f"""# Changelog — modelo antifraude

## v1.0.0 — {data}
- **Tipo:** modelo e pipeline; contrato v1.0 preservado.
- **Versão no MLflow:** antifraude v{mv.version}.
- **O que mudou:** primeiro Pipeline completo; {campeao['nome']}; corte {threshold:.2f}; monitoramento reutilizável.
- **Por quê:** menor custo na validação temporal e atendimento aos três pisos do negócio.
- **Evidência:** run `{mv.run_id}`, experimento `{wf.EXPERIMENTO}`; referência em `referencia/`; comparação em `avaliacao/`; paridade em `artefatos/`.
- **Impacto:** {brl(m['custo']/2)}/semana vs {brl(piso_regra/2)} na regra Pix; economia {brl(economia/2)}/semana ({economia/piso_regra:.1%}); recall {m['recall']:.1%}; bloqueio {m['taxa_bloqueio']:.1%}; {int(m['legitimas_bloqueadas'])} legítimas bloqueadas nas duas semanas.
- **Retreinou?** sim; somente semanas 1–10. Validação 11–12 usada para seleção, sem teste futuro independente.
- **Monitoramento:** KS D > 0,10 / TV > 0,05 com BH q < 0,01; bloqueio ±2 pp; custo por mil +25%; recall −10 pp. Rótulos fechados obrigatórios para performance.
"""
Path("CHANGELOG.md").write_text(changelog, encoding="utf-8")
with mlflow.start_run(run_id=mv.run_id):
    mlflow.log_artifacts(str(pasta_zero), "monitoramento/relatorio_zero")
    mlflow.log_text(limites_texto, "monitoramento/limites.txt")
    mlflow.log_artifact("monitoramento.py", "monitoramento")
    mlflow.log_artifact("CHANGELOG.md", "governanca")
    mlflow.log_artifact(str(wf.ARTEFATOS / "paridade_predict.csv"), "validacao_serving")
    mlflow.log_artifact(str(wf.ARTEFATOS / "paridade_lotes.json"), "validacao_serving")
wf.salvar_json(wf.ARTEFATOS / "resultado.json", {"modelo": campeao["nome"], "versao": mv.version,
    "run_id": mv.run_id, "threshold": threshold, "metricas": m, "economia_semanal": economia/2,
    "custos_baseline": {"aprovar_tudo": piso_aprovar, "regra_pix": piso_regra,
                       "proprio_modelo_05": campeao["padrao"]["custo"]}})
# Registra uma decisão histórica sem duplicá-la em reexecuções da mesma versão.
decisoes = requests.get(f"{config.SISTEMA_URL}/api/decisoes", timeout=30).json()
diagnostico = f"Baseline parte 1: {len(zero['alertas'])} alertas no relatório zero; versão {mv.version}."
if not any(d["diagnostico"] == diagnostico for d in decisoes):
    r = requests.post(f"{config.SISTEMA_URL}/api/decisoes", json={"semana": 12,
        "diagnostico": diagnostico, "acao": "Manter champion e investigar custo por canal; acompanhar a próxima semana com rótulos fechados.",
        "retreinou": False, "observacoes": "Validação histórica; sem novo treino após promoção. " + limites_texto}, timeout=30)
    r.raise_for_status()
print("Changelog, evidências e baseline vinculados à run champion.")
''')
texto(35, '''
Os limites e a análise do relatório zero estão justificados acima. A execução
salva nesta célula a interpretação quantitativa antes de empacotar a entrega.
''')
nb["cells"][35]["metadata"]["resposta"] = "limites"
codigo(37, '''
# Nome inferido do diretório pessoal do workspace; ajuste se necessário.
NOME_ALUNO = "Miguel Araujo"
from scripts.empacotar import empacotar
empacotar(NOME_ALUNO)
print("O executor salva todas as células e atualiza o ZIP após o término.")
''')
nb["metadata"]["kernelspec"] = {"display_name": "Python 3 (containers)", "language": "python", "name": "python3"}
p.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("Notebook preenchido; pronto para executar.")
