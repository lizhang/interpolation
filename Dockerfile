FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-6:latest

WORKDIR /app

COPY frame-interpolation/ frame-interpolation/
COPY requirements_worker.txt .
COPY worker.py .

ENV PYTHONPATH=/app/frame-interpolation

RUN pip install --no-cache-dir -r frame-interpolation/requirements.txt \
    && pip install --no-cache-dir -r requirements_worker.txt

CMD ["python", "worker.py"]