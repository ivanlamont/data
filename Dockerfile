FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install -e ".[yahoo]"

EXPOSE 8000
CMD ["ml4t-data", "server", "--host", "0.0.0.0", "--port", "8000"]
