setup:
	pip install -r requirements.txt

conda:
	python -m ipykernel install --user --name=rainbow-is-all-you-need

clean:
	git clean -xdf
