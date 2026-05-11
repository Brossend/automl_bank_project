install:
	pip install -r requirements.txt

download:
	python -m src.download_data

train:
	python -m src.train

evaluate:
	python -m src.evaluate

test:
	pytest

monitor:
	python -m src.monitor

docker-build:
	docker compose build

docker-run:
	docker compose up
