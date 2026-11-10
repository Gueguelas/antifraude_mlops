#!/bin/sh
set -eu
psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d postgres -f /sql/01_criar_bancos.sql
psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d antifraude -f /sql/02_criar_tabelas.sql
