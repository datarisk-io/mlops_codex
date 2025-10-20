# Datarisk MLOps - SDK

![ikjo](https://img.shields.io/pypi/v/datarisk-mlops-codex)

A Python SDK for interacting with the MLOps API, providing tools for training, deploying, and monitoring machine learning models.

---

## Table of Contents

- [Installation](#installation)
- [Getting Started](#example-of-usage)
- [Example of Usage](#example-of-usage)
- [Support](#support)
- [Contributing](#contributing)

---

## Installation

```bash
  pip install datarisk-mlops-codex
```

---

## Getting started

To use the SDK, you must be logged in to the application. This can be done by importing one of the provided clients, as shown in the example below
```python
from mlops_codex.model import MLOpsModelClient

client = MLOpsModelClient()
```

## Example of usage

```python
PATH = './samples/asyncModel/'

# Deploying a new model
model = client.create_model(
    model_name='Teste notebook Async',
    model_reference='score',
    source_file=PATH+'app.py',
    model_file=PATH+'model.pkl',
    requirements_file=PATH+'requirements.txt',
    schema=PATH+'schema.csv', 
    python_version='3.9',
    operation="Async",
    input_type='csv',
    group='datarisk'
)

PATH = './samples/asyncModel/'
execution = model.predict(data=PATH+'input.csv', group_token='TODO', wait_complete = False)
```


There's also some [example](https://github.com/datarisk-io/mlops_codex/tree/master/notebooks) notebooks.

---

## Support

* For help or questions, visit the [documentation](https://datarisk-io.github.io/mlops_codex)

---

## Contributing

* To learn more about making a contribution to datarisk-mlops-codex, please see our [Contributing guide](CONTRIBUTING.md).
