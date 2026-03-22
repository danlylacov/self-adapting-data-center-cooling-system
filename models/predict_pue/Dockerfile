FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY models/predict_pue/requirements_model_pue.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir -r /tmp/requirements.txt

COPY models /app/models
COPY DC_digital_twin /app/DC_digital_twin

EXPOSE 8011

CMD ["uvicorn", "models.predict_pue.api:app", "--host", "0.0.0.0", "--port", "8011"]
