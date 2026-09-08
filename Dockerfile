FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install --extra-index-url https://download.pytorch.org/whl/cpu 'torch>=2.2' && pip install .
COPY experiments ./experiments
ENTRYPOINT ["python", "-m", "cvar_psha.eq_jepa.train"]
