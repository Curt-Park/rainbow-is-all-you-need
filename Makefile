init:
	mise trust && mise install

setup:
	uv sync

run:
	marimo edit $(notebook)

lint:
	uv run ruff check .

format:
	uv run ruff format .

clean:
	git clean -xdf
