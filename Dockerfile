FROM python:3.12-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    portaudio19-dev \
    espeak \
    espeak-ng-data \
    libespeak-ng1 \
    libespeak1 \
    ffmpeg \
    git \
    cron \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt
COPY . .

COPY crontab.txt /etc/cron.d/cronjob
RUN chmod 0644 /etc/cron.d/cronjob && \
    crontab /etc/cron.d/cronjob && \
    touch /var/log/cron.log

ENV PYTHONUNBUFFERED=1

EXPOSE 8501
CMD cron && streamlit run bot.py --server.port=8501 --server.address=0.0.0.0
