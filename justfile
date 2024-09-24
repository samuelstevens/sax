docs: lint
    uv run pdoc3 --force --output-dir docs/md --config latex_math=True get_activations
    uv run pdoc3 --force --html --output-dir docs/html --config latex_math=True get_activations

test: lint
	pytest

lint: fmt
    ruff check .

fmt:
    isort .
    ruff format --preview .
