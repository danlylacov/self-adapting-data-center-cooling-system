FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY models/predict_load/requirements_model_load.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir -r /tmp/requirements.txt

COPY models /app/models

EXPOSE 8010

CMD ["uvicorn", "models.predict_load.api:app", "--host", "0.0.0.0", "--port", "8010"]
