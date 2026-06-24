PYTHON ?= python

.PHONY: help install test check desktop server client server-bat

help:
	@echo "make install | test | check | desktop | server | client | server-bat"

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

check:
	$(PYTHON) -m compileall app tests

lint:
	$(PYTHON) -m ruff check app tests

format:
	$(PYTHON) -m ruff format app tests

desktop:
	$(PYTHON) -m app.desktop

server:
	$(PYTHON) -m app.main

client:
	cmd /c start_client.bat

server-bat:
	cmd /c start_server.bat
