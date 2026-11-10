.PHONY: setup up test executar monitorar entrega down
setup:
	python3 scripts/configurar_ambiente.py
up: setup
	docker compose up -d --build --wait
test:
	docker compose exec -T app pytest -q --junitxml=artefatos/testes.xml
executar:
	docker compose exec -T app python scripts/executar_notebook.py
monitorar:
	docker compose exec -T app python scripts/monitorar_semana.py --semana $(SEMANA)
entrega: test
	docker compose exec -T app python scripts/finalizar_entrega.py
down:
	docker compose down
