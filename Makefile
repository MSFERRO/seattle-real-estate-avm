.PHONY: help install train predict test run-api docker-build docker-run docker-stop

help:
	@echo "Available commands:"
	@echo "  make install       Install python requirements"
	@echo "  make train         Run model training and 5-fold CV benchmark"
	@echo "  make predict       Run batch predictions on future_unseen_examples.csv"
	@echo "  make test          Run pytest test suite"
	@echo "  make run-api       Start FastAPI production server locally"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-run    Run Docker container"
	@echo "  make docker-stop   Stop Docker container"

install:
	pip install -r requirements.txt

train:
	python -m src.models.train

predict:
	python -m src.models.predict

test:
	pytest -v tests/

run-api:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t seattle-house-price-api:latest .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down
