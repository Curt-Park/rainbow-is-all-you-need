init:
	mise trust && mise install

setup:
	uv sync --no-dev

setup-dev:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff format .

clean:
	git clean -xdf
