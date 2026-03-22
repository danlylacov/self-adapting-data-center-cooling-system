FROM python:3.11-slim

WORKDIR /app

COPY models/predict_ga/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY GA /app/GA
COPY GA/tuned_params.json /app/tuned_params.json
COPY models/predict_ga/app.py /app/app.py

ENV PYTHONPATH=/app
# Явный путь в образе (код по умолчанию ищет v2/GA/… относительно исходников)
ENV TUNED_PARAMS_PATH=/app/tuned_params.json

EXPOSE 8013

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8013"]
