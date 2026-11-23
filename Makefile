PY ?= python

.PHONY: install data train quick report test lint app serve space

install:
	$(PY) -m pip install -e ".[track,serve,app,dev]"

data:            ## download the dataset (~300 MB) and convert it to parquet
	ute prepare

train:           ## full run on all ~14M rows (91 minutes on a busy 14-core laptop)
	ute train

quick:           ## 10% sample, fewer bootstrap replicates
	ute train --sample-frac 0.1 --n-boot 50

report:          ## regenerate docs/results.md and the README results block
	ute report

test:
	$(PY) -m pytest

lint:
	ruff check . && ruff format --check .

app:
	streamlit run app/streamlit_app.py

serve:
	ute serve

space:           ## assemble a Hugging Face Space (Docker SDK) in dist/space
	rm -rf dist/space && mkdir -p dist/space/app
	cp app/streamlit_app.py dist/space/app/
	cp -r artifacts dist/space/app/artifacts
	cp deploy/hf-space/Dockerfile deploy/hf-space/README.md deploy/hf-space/requirements.txt dist/space/
	@echo "Next: hf upload <user>/uplift-budget-simulator dist/space . --repo-type space"
