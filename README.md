# Banco Aurora — Prevenção a Fraudes · Prática de MLOps

Este é o sistema de produção da prática. Ele recebe os lotes semanais de transações, valida cada lote contra o contrato de features, pontua com o modelo que está no MLflow como `@champion` e mostra ao negócio quanto a fraude está custando.

O **modelo é seu**. O sistema só sabe usar o que você registrar no MLflow.

## Entrega implementada — parte 1

O notebook `parte1.ipynb` está preenchido e executado, com as respostas em texto,
gráficos, comparação das runs e evidências de produção. O nome de entrega foi
preenchido como **Miguel Araujo**, inferido do workspace; ajuste `NOME_ALUNO` se necessário.

| Alternativa | Custo médio por semana (validação 11–12) |
|---|---:|
| Aprovar tudo | R$ 188.017,77 |
| Regra Pix > R$ 1.000 | R$ 112.674,90 |
| Regressão logística, corte 0,70 | R$ 18.148,53 |
| Boosting, corte padrão 0,50 | R$ 20.810,62 |
| **Boosting, corte escolhido 0,06** | **R$ 12.424,30** |

Economia estimada de **R$ 100.250,59 por semana (89,0%)** frente à regra Pix.
No total de 39.497 transações de validação, o campeão bloqueia 489 transações
legítimas e deixa passar 91 fraudes; recall 84,83%, precision 51,00% e PR-AUC
(average precision) 0,8363. Não houve ajuste com semanas posteriores ao histórico.
As duas semanas de validação também escolheram o modelo e o corte; esses números
não são um teste independente nem garantia de resultado futuro.

### Rodar no OrbStack

```bash
docker context use orbstack
make up          # cria .env, faz build e espera os serviços ficarem prontos
make executar    # treina, registra, promove, valida e gera notebook/HTML/ZIP
make test        # testes de contrato, custo, monitoramento e integração real
make entrega     # atualiza ZIP sem treinar nem mudar o champion
```

O Postgres é exclusivo deste projeto; não reutiliza nem modifica outros containers.
Há dois volumes persistentes (bancos e artefatos do MLflow). `make down` para a stack
preservando os volumes; `make up` volta a subir. Não use `down -v` se quiser preservar
o histórico. Reexecutar o notebook cria novas runs e uma nova versão registrada;
o alias é atualizado e as semanas históricas 11–12 são repontuadas, mantendo o histórico.
As semanas 1–10 ficam só no treino, não no painel de demonstração.

| Serviço | Endereço |
|---|---|
| Painel | http://127.0.0.1:8000 |
| Modelo em produção | http://127.0.0.1:8000/modelo |
| MLflow | http://127.0.0.1:5000 |
| Jupyter Lab | http://127.0.0.1:8888 |
| Postgres | 127.0.0.1:54321 |

Os serviços são publicados só em loopback. O token do Jupyter está no `.env`
local (não incluído na entrega); use-o na tela de login. Senha e token são gerados
uma vez por `make setup`. `requirements-lock.txt` registra as versões resolvidas e
é usado como constraints nos próximos builds. A imagem usa Python 3.12 e as versões
do enunciado para MLflow e scikit-learn.

Se a rede inspeciona TLS, o build usa o certificado indicado em `SSL_CERT_FILE`
como secret do BuildKit (fallback `/etc/ssl/cert.pem`). Nenhuma validação TLS é
desabilitada. Em outra máquina, aponte essa variável para seu bundle de CAs confiáveis.
O secret não é incorporado à imagem nem ao ZIP. Os artefatos são enviados e lidos
pelo proxy HTTP do MLflow, com volume próprio; não dependem de caminhos de disco
compartilhados entre o host e os clientes. Ver
[documentação oficial de artefatos do MLflow 2.19](https://mlflow.org/docs/2.19.0/tracking/artifacts-stores.html).

### Monitorar as próximas semanas

1. Receba o CSV em **Lotes**: o sistema valida e pontua as 13 features.
2. Rode o monitoramento imediatamente, sem rótulos:
   `make monitorar SEMANA=13`.
3. Na semana seguinte, receba os rótulos do lote anterior em **Rótulos**.
   Depois da confirmação do fechamento e cobertura integral, rode:
   `docker compose exec -T app python scripts/monitorar_semana.py --semana 13 --rotulos-fechados`.
4. Leia as tabelas em `artefatos/monitoramento/semana_13/`, investigue os alertas e
   registre a ação em **Decisões**. Não retreine só porque um teste acusou drift.

O monitor sempre baixa a referência da run do champion. Confere versão, contrato,
threshold e limites, e rejeita predições incompatíveis. Código de saída **0**:
sem alertas; **1**: investigar alertas; **2**: falha operacional (também salva `falha.json`).
Uma semana ainda não recebida é falha, não resultado saudável. A execução não é
agendada automaticamente: rode-a a cada recebimento de transações e de rótulos.

Os limites aprovados são KS D > 0,10 ou variação total > 0,05 com BH q < 0,01;
mudança absoluta de bloqueio > 2 pp; aumento do custo por mil > 25%; queda de
recall > 10 pp. Drift exige ao menos 500 observações; os alertas de performance
exigem também ao menos 10 fraudes no segmento. Qualidade do contrato é crítica.
Sem fechamento explícito e cobertura integral, recall, precision, PR-AUC e custo
ficam pendentes: rótulo ausente nunca é convertido em transação legítima.

O relatório zero **não é artificialmente limpo**: idade da conta tem p ≈ 1,13e−30,
mas D ≈ 0,059 < 0,10; não dispara drift. O custo por mil cresce cerca de 44,8%
da semana 11 para 12, gerando alertas globais e em dois canais, principalmente por
maior valor das fraudes aprovadas. A análise e a decisão de investigar sem
retreinar estão no notebook e no sistema. P-valores pequenos não bastam para
demonstrar mudança relevante; poucas semanas e clientes recorrentes limitam a inferência.

### Arquivos para avaliação

- `entrega_parte1_miguel_araujo.zip`: pacote final, com manifesto SHA-256.
- `parte1.ipynb`: roteiro respondido, executado e com saídas salvas.
- `artefatos/parte1.html`: leitura do notebook sem Jupyter.
- `entrega/evidencias_mlflow.json`: campeão, versão, run, parâmetros, métricas e saúde.
- `entrega/modelo_exportado/`: modelo completo, código customizado e dependências.
- `artefatos/referencia/` e `artefatos/relatorio_zero/`: baseline e tabelas de monitoramento.
- `artefatos/paridade_predict.csv` e `artefatos/paridade_lotes.json`: paridade da API e
  dos 39.497 scores completos dos lotes. `/predict` serializa seis casas decimais;
  comparamos exatamente essa saída e, nos lotes, a precisão completa com tolerância 1e−15.
- `artefatos/testes.xml`: resultado dos testes automatizados.

Foram aprovados 27 testes, incluindo o carregamento do modelo exportado em um
processo Python isolado, fora do workspace e sem acesso ao YAML local.

O ZIP inclui código, app original, contrato, histórico, Docker/Compose, dependências,
notebook, modelo exportado e evidências. Exclui `.env`, tokens e senhas locais.
Para reconstruir em outro computador, extraia o ZIP e rode `make up` e `make executar`.
Não é necessário conectar-se ao MLflow original para inspecionar o modelo exportado.

O procedimento manual original está preservado abaixo como alternativa ao Compose.

## O que você precisa

- **Python 3.10 a 3.13**
- **O seu Postgres** (o container que você já usa nas aulas), com um usuário que possa criar bancos
- `psql` para rodar os scripts `.sql` (pode ser o do próprio container: `docker exec -i <container> psql ...`)

Ambiente virtual é **recomendado, não obrigatório**: organize a pasta como preferir. As versões do `requirements.txt` são as testadas. Se você instalar no Python global e houver conflito com outro projeto, o ajuste é seu.

## Passo a passo

### 1. Bancos e tabelas

Um servidor Postgres, dois bancos, como nas aulas: `antifraude` (o sistema) e `mlflow` (o MLflow).

```bash
psql -h localhost -U postgres -f sql/01_criar_bancos.sql
psql -h localhost -U postgres -d antifraude -f sql/02_criar_tabelas.sql
psql -h localhost -U postgres -d antifraude -c "\dt"     # tem que listar 6 tabelas
```

Se o Postgres estiver só dentro do container, sem a porta exposta:

```bash
docker exec -i <container> psql -U postgres < sql/01_criar_bancos.sql
docker exec -i <container> psql -U postgres -d antifraude < sql/02_criar_tabelas.sql
```

O `02_criar_tabelas.sql` é idempotente: rodar duas vezes não quebra nada.

### 2. Configuração

```bash
cp .env.exemplo .env          # ajuste usuário, senha, host e porta do SEU Postgres
pip install -r requirements.txt
python config.py              # confere as conexões e imprime o comando do MLflow
```

### 3. MLflow (terminal 1)

O servidor do MLflow **não lê o `.env`**: a conexão vai na linha de comando. Use exatamente o comando que o `python config.py` imprimiu. Com os valores padrão:

```bash
python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri postgresql://postgres:1234@localhost:5432/mlflow --default-artifact-root ./mlruns
```

Abra http://127.0.0.1:5000.

Se aparecer `ImportError: cannot import name 'FallbackAsyncAdaptedQueuePool'`, rode `pip install "sqlalchemy<2.1"`.

### 4. Sistema (terminal 2)

Sempre a partir desta pasta (a raiz do projeto):

```bash
python -m uvicorn app.main:app --reload --port 8000
```

Abra http://127.0.0.1:8000. Antes do seu primeiro modelo, o topo da tela mostra **"sem modelo em produção"** — é o esperado.

### 5. Notebook (terminal 3)

```bash
jupyter notebook parte1.ipynb
```

Rode o notebook **a partir desta pasta**, com o MLflow e o sistema no ar.

## Como o modelo entra em produção

O sistema usa sempre `models:/antifraude@champion`. Para isso, a versão registrada precisa:

1. **Receber as 13 features do contrato, cruas, na ordem do contrato.** Todo o pré-processamento mora dentro do modelo (um `Pipeline`). O sistema não transforma nada.
2. **Ter `predict_proba`.** Registre com `mlflow.sklearn.log_model`. Se o seu pipeline usa código seu (uma classe ou função do `features.py`), passe `code_paths=["features.py"]`, senão o sistema não consegue carregar o modelo.
3. **Ter a tag `threshold` na VERSÃO** (não na run), com o corte de bloqueio: `score >= threshold` bloqueia.
4. Opcional, mas cobrado na rastreabilidade: tag `motivo` na versão e uma descrição. O sistema anota o motivo em **Modelo → Histórico de trocas**.

Trocar o modelo em produção = mover o alias `champion` no MLflow. O sistema percebe na próxima chamada; não precisa reiniciar. Para aplicar o modelo novo a semanas já recebidas, use **Repontuar** na tela de Lotes.

## O que tem na pasta

| Arquivo | O que é |
|---|---|
| `contrato/contrato_features.yaml` | O contrato de features v1.0. O sistema valida todo lote contra ele |
| `dados/` | Os dados da parte 1 (histórico rotulado, semanas 1 a 12) |
| `demanda_parte1.md` | O que o negócio pediu |
| `parte1.ipynb` | O roteiro técnico da parte 1: instruções, o código é seu |
| `features.py`, `monitoramento.py` | Vazios: você escreve. Serão reutilizados nas partes 2 e 3 |
| `CHANGELOG.md` | O histórico de versões do seu modelo, mantido por você |
| `app/` | O sistema: API em FastAPI e front em HTML/CSS/JS puro (`app/static/`). Encanamento: vem pronto, não precisa mexer |
| `sql/` | Criação dos bancos e das tabelas do sistema |
| `config.py`, `.env.exemplo` | Configuração das conexões |

## Como o sistema é montado

- **Back:** FastAPI (`app/main.py`), só API — devolve JSON e CSV.
- **Front:** HTML, CSS e JavaScript puros (`app/static/`), sem biblioteca. Cada página chama a API com `fetch`.
- O próprio FastAPI serve as páginas. A documentação automática das rotas fica em http://127.0.0.1:8000/docs.

| Página | O que faz | Rotas que ela chama |
|---|---|---|
| `/` | Painel do negócio, por semana: volume, % bloqueado e, quando há rótulo, fraude evitada, fraude perdida e custo | `GET /api/painel` |
| `/lotes` | Recebe o CSV de transações de uma semana; valida, grava e pontua. Também **Repontuar** e **baixar dados** | `GET/POST /api/lotes`, `POST /api/lotes/{n}/repontuar` |
| `/rotulos` | Recebe o CSV de contestações (rótulos) | `POST /api/rotulos` |
| `/modelo` | Quem está em produção, threshold e histórico de trocas | `GET /api/modelo` |
| `/decisoes` | Seu registro de decisões, por semana, e exportação | `GET/POST /api/decisoes` |
| todas | O selo do modelo no topo | `GET /api/saude` |

Outras rotas úteis:

| Rota | O que faz |
|---|---|
| `POST /predict` | Pontua transações avulsas (lista de JSON com as 13 features) |
| `GET /api/semanas/{n}/dados.csv` | Transações da semana + score vigente + rótulo (quando houver). É o que o seu monitoramento lê |
| `GET /api/painel.csv` | O painel em CSV |
| `GET /api/decisoes/exportar.csv` | Suas decisões, para a entrega das partes 2 e 3 |

O painel mostra **a visão do negócio**. Drift e análise de performance por segmento são seus, nos notebooks.

## Recomeçar do zero

```bash
psql -h localhost -U postgres -c "DROP DATABASE antifraude" -c "DROP DATABASE mlflow"
```

Depois apague a pasta `mlruns/` e refaça o passo 1.
