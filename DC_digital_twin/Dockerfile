FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app/DC_digital_twin
ENV PYTHONPATH=/app

COPY DC_digital_twin/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

COPY DC_digital_twin /app/DC_digital_twin

EXPOSE 8000

# Optional: путь к YAML (если файл есть). Иначе — встроенный `src/default_config.py`.
# ENV CONFIG_PATH=config/custom.yaml

# `DC_digital_twin/run_api.py` reads API_HOST/API_PORT env vars.
ENV API_HOST=0.0.0.0 \
    API_PORT=8000

CMD ["python", "run_api.py"]
