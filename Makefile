install:
	uv sync \
  		--no-install-package torchvision \
  		--no-install-package torchaudio
run:
	uv run python -m src/rag

debug:
	pdb

clean:
	find . -path ./.venv -prune -o -type d -name __pycache__ -exec rm -rf {} +
	find . -path ./.venv -prune -o -type d -name .mypy_cache -exec rm -rf {} +
