# Web app image. Runs the AI with ONNX Runtime — no PyTorch, so it is small and fast to start.
# (Training uses the Colab notebook or a GPU machine with `pip install -e ".[ml]"`.)
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[app,onnx]"
COPY app ./app
COPY examples ./examples
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
