import numpy as np
import pandas as pd
import pytest
from features import carregar_contrato
import monitoramento as mon


def test_custo_financeiro_manual_e_limite_inclusivo():
    y, s, v = [1, 0, 1, 0], [0.19, 0.20, 0.20, 0.01], [200, 80, 900, 20]
    m = mon.metricas(y, s, v, 0.20)
    assert m["custo"] == 215
    assert m["legitimas_bloqueadas"] == 1
    assert m["fraudes_aprovadas"] == 1
    assert m["recall"] == 0.5
    assert m["precision"] == 0.5


def test_threshold_economico_supera_padrao():
    df = pd.DataFrame({"fraude": [1, 0, 1, 0], "valor_transacao": [200, 80, 900, 20]})
    s = [0.2, 0.1, 0.3, 0.01]
    curva = mon.varrer_thresholds(df, s)
    assert curva.iloc[0].custo == 0
    assert curva.iloc[0].custo < mon.metricas(df.fraude, s, df.valor_transacao, 0.5)["custo"]


@pytest.mark.parametrize("y,b,v", [([np.nan], [0], [100]), ([0], [0], [np.inf]), ([2], [0], [10]), ([0], [0, 1], [10])])
def test_custo_rejeita_incompletos_e_invalidos(y, b, v):
    with pytest.raises(ValueError):
        mon.custo_decisoes(y, b, v)


@pytest.fixture
def lote():
    df = pd.read_csv("dados/transacoes_semanas_01_12.csv", nrows=1000)
    df["score"] = np.linspace(0, 1, len(df))
    df["fraude"] = (df.index % 20 == 0).astype(int)
    return df


def test_rotulos_aguardam_fechamento_explicito(lote):
    assert (mon.performance(lote, 0.2).status == "aguardando_rotulos").all()
    fechado = mon.performance(lote, 0.2, rotulos_fechados=True)
    assert (fechado.status == "completa").all()
    parcial = lote.copy()
    parcial.loc[parcial.index[0], "fraude"] = np.nan
    assert (mon.performance(parcial, 0.2, rotulos_fechados=True).status == "aguardando_rotulos").all()
    assert (mon.performance(lote.drop(columns="fraude"), 0.2).status == "aguardando_rotulos").all()


def test_identico_sem_drift_e_shift_detectado(lote):
    c = carregar_contrato()
    identico = mon.drift(lote, lote.copy(), c)
    assert not identico.alerta.any()
    assert (identico.efeito == 0).all()
    alterado = lote.copy()
    alterado["valor_transacao"] *= 100
    alterado["canal"] = "cartao_online"
    alterado["score"] = 0.99
    shift = mon.drift(lote, alterado, c).set_index("feature")
    assert shift.loc["valor_transacao", "alerta"]
    assert shift.loc["canal", "alerta"]
    assert shift.loc["score", "alerta"]


def test_qualidade_critica_independe_de_rotulos(lote):
    ruim = lote.copy()
    ruim["canal"] = "boleto"
    r = mon.relatorio(lote, ruim, carregar_contrato(), 0.2)
    assert not r["qualidade"].ok.all()
    assert r["alertas"].iloc[0].tipo == "qualidade"


def test_amostra_pequena_inconclusiva(lote):
    r = mon.relatorio(lote, lote.head(10), carregar_contrato(), 0.2)
    assert "amostra_insuficiente" in r["alertas"].tipo.values


def test_score_nulo_e_lote_vazio_sao_qualidade_critica(lote):
    ruim = lote.copy()
    ruim.loc[0, "score"] = np.nan
    for df in [ruim, lote.iloc[:0]]:
        r = mon.relatorio(lote, df, carregar_contrato(), 0.2)
        assert "qualidade" in r["alertas"].tipo.values


def test_correcao_bh():
    np.testing.assert_allclose(mon._bh([0.01, 0.04, 0.03, 0.002]), [0.02, 0.04, 0.04, 0.008])
