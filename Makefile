.PHONY: install run clean

install:
	pip install -r requirements.txt

run:
	python3 src/gui.py

clean:
        find . -name '*.pyc' -delete
        find . -name '__pycache__' -delete
