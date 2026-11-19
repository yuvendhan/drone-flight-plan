.PHONY: install test lint format app docker
install:
	pip install -e ".[all]"
test:
	pytest -q
lint:
	ruff check src tests app
format:
	ruff format src tests app
app:
	streamlit run app/streamlit_app.py
docker:
	docker build -t drone-flight-plan . && docker run -p 8501:8501 drone-flight-plan
