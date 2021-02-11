setup:
	pip install -r requirements.txt

dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	jupyter contrib nbextension install --user
	jupyter nbextensions_configurator enable --user
	python3 -m ipykernel install --user
