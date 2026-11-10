# Changelog — modelo antifraude

## v1.0.0 — 10/11/2026
- **Tipo:** modelo e pipeline; contrato v1.0 preservado.
- **Versão no MLflow:** antifraude v3.
- **O que mudou:** primeiro Pipeline completo; hist_gradient_boosting; corte 0.06; monitoramento reutilizável.
- **Por quê:** menor custo na validação temporal e atendimento aos três pisos do negócio.
- **Evidência:** run `4398e4b800f14df1a95d6cd1fec8ee64`, experimento `Aurora | Parte 1 | custo e contrato v1.0`; referência em `referencia/`; comparação em `avaliacao/`; paridade em `artefatos/`.
- **Impacto:** R$ 12.424,30/semana vs R$ 112.674,90 na regra Pix; economia R$ 100.250,59/semana (89.0%); recall 84.8%; bloqueio 2.5%; 489 legítimas bloqueadas nas duas semanas.
- **Retreinou?** sim; somente semanas 1–10. Validação 11–12 usada para seleção, sem teste futuro independente.
- **Monitoramento:** KS D > 0,10 / TV > 0,05 com BH q < 0,01; bloqueio ±2 pp; custo por mil +25%; recall −10 pp. Rótulos fechados obrigatórios para performance.

<!-- historico-publicacoes -->

## v1.0.0 — publicação anterior (MLflow v2)
- **Tipo:** modelo/pipeline; primeiro ensaio de publicação da parte 1.
- **Versão no MLflow:** antifraude v2; run `6fe2efed5e954b42b917ee5660534f89`.
- **O que mudou:** publicação inicial, depois substituída pela versão final v3 durante a conclusão das evidências e validações.
- **Por quê:** conferir integração real antes do fechamento da entrega; código original preservado na run.
- **Evidência:** métricas e artefatos da run; custo de validação R$ 24848.61, corte 0.06.
- **Impacto:** versão retirada do alias; histórico de predições e trocas preservado.
- **Retreinou?** sim, nas semanas 1–10.

## v1.0.0 — publicação anterior (MLflow v1)
- **Tipo:** modelo/pipeline; primeiro ensaio de publicação da parte 1.
- **Versão no MLflow:** antifraude v1; run `efeb27a0b3ee4f7ebd95d80e7ae65ad5`.
- **O que mudou:** publicação inicial, depois substituída pela versão final v3 durante a conclusão das evidências e validações.
- **Por quê:** conferir integração real antes do fechamento da entrega; código original preservado na run.
- **Evidência:** métricas e artefatos da run; custo de validação R$ 24848.61, corte 0.06.
- **Impacto:** versão retirada do alias; histórico de predições e trocas preservado.
- **Retreinou?** sim, nas semanas 1–10.
