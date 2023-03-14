# Neomaril Codex

## About
Package for interacting with Neomaril, a tool for deploying ML models.

## Getting started

### Intalation
```
  pip install neomaril-codex
```

### How to use

Check the [documentation](https://datarisk-io.github.io/mlops-neomaril-codex) page for more information
There also some [examples](https://github.com/datarisk-io/mlops-neomaril-codex/tree/master/notebooks) notebooks.

### For developers

Install pipenv
```
  pip install pipenv
```

Install the package enviroment
```
  pipenv update --dev
  pipenv shell
```

Publish to Pypi
```
  python setup.py sdist
  twine upload dist/*
```
