APP_NAME=main:app
HOST=127.0.0.1
PORT=8000

.PHONY: run dev install freeze lint clean

run:
	@echo "Running API at http://$(HOST):$(PORT)"
	uvicorn $(APP_NAME) --host $(HOST) --port $(PORT)

dev:
	@echo "Running with reload (development mode)"
	uvicorn $(APP_NAME) --host $(HOST) --port $(PORT) --reload

install:
	pip install -r requirements.txt

freeze:
	pip freeze > requirements.txt

lint:
	ruff .

clean:
	find . -type d -name '__pycache__' -exec rm -r {} +