# Neomaril Codex

Para mudar para a versão em inglês deste README, clique [aqui](./README.md).

## Sobre

Pacote para interagir com o Neomaril, uma ferramenta para deploy de modelos de Machine Learning (ML).

## Começando

### Instalação

```
  pip install neomaril-codex
```

### Como usar

Leia a [documentação](https://datarisk-io.github.io/mlops-neomaril-codex) para mais informações.

Disponibilizamos também alguns notebooks de [exemplo](https://github.com/datarisk-io/mlops-neomaril-codex/tree/master/notebooks).

### Para desenvolvedores

Instale o pipenv

```
  pip install pipenv
```

Instale o ambiente do pacote

```
  pipenv update --dev
  pipenv shell
```

Publique no Pypi

```
  python setup.py sdist
  twine upload dist/*
```
