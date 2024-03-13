FROM python:3.10-slim

WORKDIR /app

RUN apt update \
    && pip install --upgrade pip setuptools
COPY . .

RUN pip install -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/app/"

ENTRYPOINT ["/bin/sh", "-c"]
CMD ["python trainer/main.py"]