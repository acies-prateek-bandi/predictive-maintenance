ARG PYTHON_VERSION=3.10.6
FROM python:${PYTHON_VERSION}-slim as base
WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt

CMD ["python3", "./app.py"]