FROM python:3.11-slim

WORKDIR /app

# install deps first so layer is cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p embeddings clustering/artifacts data

EXPOSE 8000

ENTRYPOINT ["bash", "entrypoint.sh"]
