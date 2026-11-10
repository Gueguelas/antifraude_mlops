FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MPLBACKEND=Agg
WORKDIR /workspace
COPY requirements*.txt ./
RUN --mount=type=secret,id=build_ca \
    if [ -f requirements-lock.txt ]; then \
      PIP_CERT=/run/secrets/build_ca pip install -r requirements.txt -r requirements-dev.txt -c requirements-lock.txt; \
    else PIP_CERT=/run/secrets/build_ca pip install -r requirements.txt -r requirements-dev.txt; fi
COPY . .
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
