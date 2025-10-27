from enum import StrEnum


class ModelType(StrEnum):
    CLASSIFICATION = 'Classification'
    REGRESSION = 'Regression'
    UNSUPERVISED = 'Unsupervised'


def is_valid_model_type(v: str):
    """
    Check whether the given value is a valid model type

    Args:
        v (str): Unvalidated model type
    """
    try:
        return ModelType(v.capitalize()).value
    except ValueError:
        valid = ', '.join(ModelType.__members__.values())
        raise ValueError(f'Invalid model type. Valid types are: {valid}')
