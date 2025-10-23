from enum import StrEnum


class ModelType(StrEnum):
    CLASSIFICATION = 'Classification'
    REGRESSION = 'Regression'
    UNSUPERVISED = 'Unsupervised'


def is_valid_model_type(value: str):
    try:
        return ModelType(value).value
    except ValueError:
        valid = ', '.join(ModelType.__members__.values())
        raise ValueError(f'Invalid model type. Valid types are: {valid}')
