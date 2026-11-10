"""Qualidade, drift, saída e performance com fechamento explícito dos rótulos."""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.metrics import average_precision_score, precision_score, recall_score

from features import validar_features

CUSTO_FP = 15.0
LIMITES = {"ks_d": 0.10, "tv": 0.05, "q": 0.01, "min_n": 500,
           "bloqueio_delta": 0.02, "custo_aumento": 0.25, "recall_queda": 0.10}


def _arrays(fraude, bloqueada, valores):
    y, b, v = np.asarray(fraude), np.asarray(bloqueada), np.asarray(valores, dtype=float)
    if y.ndim != 1 or b.shape != y.shape or v.shape != y.shape or not len(y):
        raise ValueError("Vetores devem ter mesmo comprimento não nulo.")
    if not np.isin(y, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise ValueError("Rótulos e decisões devem ser binários e completos.")
    if not np.isfinite(v).all() or (v <= 0).any():
        raise ValueError("Valores devem ser positivos e finitos.")
    return y.astype(bool), b.astype(bool), v


def custo_decisoes(fraude, bloqueada, valores):
    y, b, v = _arrays(fraude, bloqueada, valores)
    return float(v[y & ~b].sum() + CUSTO_FP * (~y & b).sum())


def metricas(fraude, scores, valores, threshold):
    s = np.asarray(scores, dtype=float)
    if not np.isfinite(s).all() or ((s < 0) | (s > 1)).any():
        raise ValueError("Scores devem ser probabilidades finitas em [0, 1].")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold fora de [0, 1].")
    y, b, v = _arrays(fraude, s >= threshold, valores)
    fn, fp = y & ~b, ~y & b
    custo = custo_decisoes(y, b, v)
    return {"n": len(y), "fraudes": int(y.sum()), "taxa_fraude": float(y.mean()),
            "recall": float(recall_score(y, b, zero_division=0)) if y.any() else np.nan,
            "precision": float(precision_score(y, b, zero_division=0)),
            "pr_auc": float(average_precision_score(y, s)) if y.any() else np.nan,
            "custo": custo, "custo_por_1000": custo / len(y) * 1000,
            "legitimas_bloqueadas": int(fp.sum()), "fraudes_aprovadas": int(fn.sum()),
            "fraude_perdida": float(v[fn].sum()), "fraude_evitada": float(v[y & b].sum()),
            "taxa_bloqueio": float(b.mean()), "valor_bloqueado": float(v[b].sum())}


def varrer_thresholds(df, scores, thresholds=None):
    thresholds = np.arange(1, 100) / 100 if thresholds is None else thresholds
    linhas = [{"threshold": float(t), **metricas(df.fraude, scores, df.valor_transacao, t)}
              for t in thresholds]
    return pd.DataFrame(linhas).sort_values(["custo", "legitimas_bloqueadas", "threshold"]).reset_index(drop=True)


def _bh(p):
    """Benjamini-Hochberg para a família de testes da semana."""
    p = np.asarray(p, dtype=float)
    ordem = np.argsort(p)
    q = np.minimum.accumulate((p[ordem] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    saida = np.empty_like(p)
    saida[ordem] = np.minimum(q, 1)
    return saida


def drift(referencia, lote, contrato):
    linhas = []
    for f in contrato["features"] + [{"nome": "score", "tipo": "float"}]:
        c = f["nome"]
        if c not in referencia or c not in lote:
            raise ValueError(f"Coluna ausente para drift: {c}")
        a, b = referencia[c].dropna(), lote[c].dropna()
        if not len(a) or not len(b):
            linhas.append({"feature": c, "teste": "sem_dados", "estatistica": np.nan,
                           "p_valor": 1.0, "efeito": np.nan, "limite": 0.0,
                           "n_ref": len(a), "n_lote": len(b)})
            continue
        categorica = f["tipo"] in ("categoria", "bool") or c in ("hora_transacao", "dia_semana")
        if categorica:
            cats = sorted(set(a.astype(str)) | set(b.astype(str)))
            cont = np.array([a.astype(str).value_counts().reindex(cats, fill_value=0),
                             b.astype(str).value_counts().reindex(cats, fill_value=0)])
            chi, p = chi2_contingency(cont, correction=False)[:2] if len(cats) > 1 else (0.0, 1.0)
            efeito = 0.5 * np.abs(cont[0] / len(a) - cont[1] / len(b)).sum()
            teste, estatistica, limite = "qui_quadrado", float(chi), LIMITES["tv"]
        else:
            ks = ks_2samp(a.astype(float), b.astype(float), method="asymp")
            estatistica, p, efeito = float(ks.statistic), float(ks.pvalue), float(ks.statistic)
            teste, limite = "ks", LIMITES["ks_d"]
        linhas.append({"feature": c, "teste": teste, "estatistica": estatistica,
                       "p_valor": float(p), "efeito": float(efeito), "limite": limite,
                       "n_ref": len(a), "n_lote": len(b)})
    tabela = pd.DataFrame(linhas)
    tabela["q_valor"] = _bh(tabela.p_valor)
    tabela["amostra_suficiente"] = tabela[["n_ref", "n_lote"]].min(axis=1) >= LIMITES["min_n"]
    tabela["alerta"] = ((tabela.efeito > tabela.limite) & (tabela.q_valor < LIMITES["q"])
                         & tabela.amostra_suficiente) | (tabela.teste == "sem_dados")
    return tabela


def saida_modelo(lote, threshold):
    if not len(lote):
        raise ValueError("Lote vazio.")
    scores = pd.to_numeric(lote.score, errors="coerce")
    if scores.isna().any() or not np.isfinite(scores).all() or not scores.between(0, 1).all():
        raise ValueError("Score inválido.")
    b = scores >= threshold
    return pd.DataFrame([{"n": len(lote), "taxa_bloqueio": float(b.mean()),
                          "valor_bloqueado": float(lote.loc[b, "valor_transacao"].sum()),
                          "score_medio": float(scores.mean())}])


def performance(lote, threshold, segmento="canal", rotulos_fechados=False):
    """Rótulo ausente nunca vira legítimo; aguarda fechamento e cobertura total."""
    integral = "fraude" in lote and lote.fraude.notna().all()
    grupos = [("global", lote)] + [(str(k), g) for k, g in lote.groupby(segmento, dropna=False)]
    linhas = []
    for nome, g in grupos:
        cobertura = float(g.fraude.notna().mean()) if "fraude" in g else 0.0
        pronta = rotulos_fechados and integral and cobertura == 1.0
        m = metricas(g.fraude, g.score, g.valor_transacao, threshold) if pronta else {"n": len(g)}
        linhas.append({"segmento": nome, "rotulos_fechados": rotulos_fechados,
                       "cobertura_rotulos": cobertura, "status": "completa" if pronta else "aguardando_rotulos", **m})
    return pd.DataFrame(linhas)


def relatorio(referencia, lote, contrato, threshold, rotulos_fechados=False, segmento="canal"):
    qualidade = []
    for nome, df in [("referencia", referencia), ("lote", lote)]:
        try:
            validar_features(df, contrato, estrito=False)
            saida_modelo(df, threshold)
            qualidade.append({"origem": nome, "ok": True, "detalhe": "contrato válido"})
        except ValueError as e:
            qualidade.append({"origem": nome, "ok": False, "detalhe": str(e)})
    qualidade = pd.DataFrame(qualidade)
    if not qualidade.ok.all():
        return {"qualidade": qualidade, "alertas": pd.DataFrame([
            {"tipo": "qualidade", "segmento": "global", "acao": "Interromper análise e corrigir contrato/dado."}])}
    d = drift(referencia, lote, contrato)
    saida = saida_modelo(lote, threshold)
    saida_ref = saida_modelo(referencia, threshold)
    saida["delta_taxa_bloqueio"] = saida.taxa_bloqueio - saida_ref.taxa_bloqueio.iloc[0]
    perf = performance(lote, threshold, segmento, rotulos_fechados)
    perf_ref = performance(referencia, threshold, segmento, True)
    alertas = [{"tipo": "drift", "segmento": r.feature, "acao": "Investigar origem e impacto; sem retreino automático."}
               for r in d[d.alerta].itertuples()]
    if len(lote) < LIMITES["min_n"]:
        alertas.append({"tipo": "amostra_insuficiente", "segmento": "global", "acao": "Acumular observações; não concluir ausência de drift."})
    if abs(saida.delta_taxa_bloqueio.iloc[0]) > LIMITES["bloqueio_delta"]:
        alertas.append({"tipo": "saida", "segmento": "global", "acao": "Investigar variação de bloqueio acima de 2 pontos percentuais."})
    for r in perf[perf.status == "completa"].itertuples():
        ant = perf_ref[perf_ref.segmento == r.segmento]
        if ant.empty or r.fraudes < 10 or r.n < LIMITES["min_n"]:
            continue
        ant = ant.iloc[0]
        if r.custo_por_1000 > ant.custo_por_1000 * (1 + LIMITES["custo_aumento"]):
            alertas.append({"tipo": "custo", "segmento": r.segmento, "acao": "Investigar aumento de custo por mil acima de 25%."})
        if r.recall < ant.recall - LIMITES["recall_queda"]:
            alertas.append({"tipo": "recall", "segmento": r.segmento, "acao": "Investigar queda de recall superior a 10 pontos percentuais."})
    return {"qualidade": qualidade, "drift": d, "saida": saida, "performance": perf,
            "alertas": pd.DataFrame(alertas, columns=["tipo", "segmento", "acao"])}
