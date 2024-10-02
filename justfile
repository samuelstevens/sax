docs: lint
    uv run pdoc3 --force --html --output-dir docs --config latex_math=True get_activations launch sax

test: lint
	uv run pytest tests

lint: fmt
    ruff check get_activations.py launch.py sax/

fmt:
    isort .
    ruff format --preview .
