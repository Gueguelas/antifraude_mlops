import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from features import carregar_contrato, colunas_contrato, criar_pipeline, validar_features


@pytest.fixture
def dados():
    c = carregar_contrato()
    return c, pd.read_csv("dados/transacoes_semanas_01_12.csv", nrows=12)[colunas_contrato(c)[0]]


@pytest.mark.parametrize("alteracao", ["ausente", "extra", "ordem", "inteiro", "infinito", "nulo", "categoria", "minimo", "bool"])
def test_contrato_rejeita_entrada_invalida(dados, alteracao):
    c, x = dados
    if alteracao == "ausente":
        x = x.drop(columns="canal")
    elif alteracao == "extra":
        x["fraude"] = 0
    elif alteracao == "ordem":
        x = x[x.columns[::-1]]
    elif alteracao == "inteiro":
        x["hora_transacao"] = 1.5
    elif alteracao == "infinito":
        x["valor_transacao"] = np.inf
    elif alteracao == "nulo":
        x["valor_transacao"] = np.nan
    elif alteracao == "categoria":
        x["canal"] = "boleto"
    elif alteracao == "minimo":
        x["valor_transacao"] = 0
    elif alteracao == "bool":
        x["destinatario_novo"] = 2
    with pytest.raises(ValueError):
        validar_features(x, c)


def test_pipeline_exportado_aceita_categoria_conhecida_ausente_treino(dados, tmp_path, monkeypatch):
    c, x = dados
    x["uf_cliente"] = "SP"
    modelo = criar_pipeline(LogisticRegression(max_iter=300), c).fit(x, [0, 1] * 6)
    exemplo = x.iloc[:2].copy()
    exemplo["uf_cliente"] = "AC"
    esperado = modelo.predict_proba(exemplo)
    caminho = tmp_path / "modelo.joblib"
    joblib.dump(modelo, caminho)
    monkeypatch.chdir(tmp_path)
    restaurado = joblib.load(caminho)
    np.testing.assert_array_equal(restaurado.predict_proba(exemplo), esperado)
    assert restaurado.n_features_in_ == 13


def test_razao_nao_recalculada_ou_agregada(dados):
    c, x = dados
    # A validação do modelo preserva o valor que a Plataforma forneceu.
    assert validar_features(x, c).razao_valor_media.equals(x.razao_valor_media)
