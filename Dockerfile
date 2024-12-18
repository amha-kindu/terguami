FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch==2.1.2 --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7000", "--log-level", "info"]
