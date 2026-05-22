FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install .

WORKDIR /app/src

CMD ["python", "main.py"]
