# Hugging Face Space (Docker SDK): builds the app straight from the GitHub repo.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HOME=/home/user
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user
ARG REPO=https://github.com/yuvendhan/drone-flight-plan.git
RUN git clone --depth 1 "$REPO" /app && pip install --upgrade pip && pip install "/app[app,onnx]" \
    && chown -R user /app
USER user
WORKDIR /app
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--server.enableXsrfProtection=false"]
