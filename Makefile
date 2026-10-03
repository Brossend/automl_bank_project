install:
	poetry install

download:
	poetry run python -m src.download_data

train:
	poetry run python -m src.train

evaluate:
	poetry run python -m src.evaluate

test:
	poetry run pytest

monitor:
	poetry run python -m src.monitor

lint:
	poetry run ruff check .
	poetry run ruff format --check .

format:
	poetry run ruff check --fix .
	poetry run ruff format .

typecheck:
	poetry run mypy src

pre-commit:
	poetry run pre-commit run --all-files

docker-build:
	docker compose build

docker-run:
	docker compose up
