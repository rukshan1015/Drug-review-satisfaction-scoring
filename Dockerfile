FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential gcc libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ui ./ui	

EXPOSE 7860

CMD ["python", "ui/infer_rating.py"]