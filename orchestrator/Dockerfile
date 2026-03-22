FROM python:3.11-slim

WORKDIR /app
COPY orchestrator/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY orchestrator /app/orchestrator

ENV PYTHONPATH=/app
EXPOSE 8030

CMD ["uvicorn", "orchestrator.app:app", "--host", "0.0.0.0", "--port", "8030"]
