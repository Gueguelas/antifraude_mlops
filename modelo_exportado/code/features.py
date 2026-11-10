"""Pré-processamento compartilhado; contrato incorporado ao modelo exportado."""
from pathlib import Path
import copy

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def carregar_contrato(caminho="contrato/contrato_features.yaml"):
    return yaml.safe_load(Path(caminho).read_text(encoding="utf-8"))


def colunas_contrato(contrato):
    nomes = [f["nome"] for f in contrato["features"]]
    categoricas = [f["nome"] for f in contrato["features"] if f["tipo"] == "categoria"]
    return nomes, categoricas, [c for c in nomes if c not in categoricas]


def validar_features(df, contrato, estrito=True):
    """Valida antes de converter: nunca trunca inteiros inválidos."""
    nomes, _, _ = colunas_contrato(contrato)
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Entrada deve ser DataFrame com as 13 features nomeadas.")
    if df.empty:
        raise ValueError("Entrada não pode ser vazia.")
    if df.columns.duplicated().any():
        raise ValueError("Colunas duplicadas.")
    if any(c not in df for c in nomes):
        raise ValueError("Features obrigatórias ausentes.")
    if estrito and list(df.columns) != nomes:
        raise ValueError("Entrada deve conter exatamente as features na ordem do contrato.")
    saida = df[nomes].copy()
    erros = []
    for f in contrato["features"]:
        c = f["nome"]
        s = saida[c]
        if s.isna().any():
            erros.append(f"{c}: nulos")
        if f["tipo"] == "categoria":
            if not s.isin(f["valores"]).all():
                erros.append(f"{c}: categoria fora do contrato")
            continue
        n = pd.to_numeric(s, errors="coerce")
        if not np.isfinite(n.to_numpy(dtype=float)).all():
            erros.append(f"{c}: valor não numérico ou infinito")
            continue
        if f["tipo"] in ("int", "bool") and ((n % 1) != 0).any():
            erros.append(f"{c}: esperado inteiro")
        if f["tipo"] == "bool" and not n.isin([0, 1]).all():
            erros.append(f"{c}: esperado 0 ou 1")
        if "min" in f and (n < f["min"]).any():
            erros.append(f"{c}: abaixo do mínimo")
        if "max" in f and (n > f["max"]).any():
            erros.append(f"{c}: acima do máximo")
        saida[c] = n
    if erros:
        raise ValueError("; ".join(erros))
    for f in contrato["features"]:
        if f["tipo"] in ("int", "bool"):
            saida[f["nome"]] = saida[f["nome"]].astype("int64")
        elif f["tipo"] == "float":
            saida[f["nome"]] = saida[f["nome"]].astype("float64")
    return saida


class ValidarContrato(TransformerMixin, BaseEstimator):
    """Guarda a especificação no estimador serializado, sem YAML externo."""
    def __init__(self, contrato):
        self.contrato = contrato

    def fit(self, X, y=None):
        validar_features(X, self.contrato)
        self.feature_names_in_ = np.array(colunas_contrato(self.contrato)[0], dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        return validar_features(X, self.contrato)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


def criar_pipeline(estimador, contrato):
    """Log1p/escala nas numéricas; one-hot com domínio inteiro do YAML.

    Nenhuma agregação é recalculada usando o futuro. IDs, semana e rótulos
    são excluídos. Escala e parâmetros são ajustados só nas semanas 1–10.
    """
    _, cats, nums = colunas_contrato(contrato)
    dominios = [f["valores"] for f in contrato["features"] if f["tipo"] == "categoria"]
    numerico = Pipeline([
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("escala", StandardScaler()),
    ])
    preprocessador = ColumnTransformer([
        ("numericas", numerico, nums),
        ("categoricas", OneHotEncoder(categories=dominios, handle_unknown="error",
                                     sparse_output=False), cats),
    ], remainder="drop", verbose_feature_names_out=True)
    return Pipeline([
        ("contrato", ValidarContrato(copy.deepcopy(contrato))),
        ("preprocessamento", preprocessador),
        ("classificador", estimador),
    ])
