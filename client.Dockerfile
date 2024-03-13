FROM python:3.10-slim

WORKDIR /app

RUN apt update \
    && pip install --upgrade pip setuptools
COPY . .

RUN pip install -r requirements.txt

EXPOSE 8080

ENTRYPOINT ["/bin/sh", "-c"]
CMD ["uvicorn client.app:app --host 0.0.0.0 --port 8080 --log-level debug"]