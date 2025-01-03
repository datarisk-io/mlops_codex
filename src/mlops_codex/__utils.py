import io
import json
import typing
from functools import wraps
from typing import Callable, Type

import yaml


def parse_dict_or_file(obj):
    if isinstance(obj, str):
        schema_file = open(obj, "rb")
    elif isinstance(obj, dict):
        schema_file = io.StringIO()
        json.dump(obj, schema_file).seek(0)

    return schema_file


def parse_url(url):
    if url.endswith("/"):
        url = url[:-1]

    if not url.endswith("/api"):
        url = url + "/api"
    return url


def parse_json_to_yaml(data) -> str:
    """Parse a loaded json as dict to yaml format

    Args:
        data (dict): data in a json format

    Returns:
        str: data in the yaml format
    """
    return yaml.dump(data, allow_unicode=True, default_flow_style=False)


def validate_kwargs(model: Type) -> Callable:
    """
    Decorator to validate keyword arguments against a TypedDict.

    Args:
        model (Type): The Type class used for validation.

    Returns:
        Callable: The decorated method with validation applied.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, **kwargs):
            missing_keys = [
                field
                for field, field_type in model.__annotations__.items()
                if type(None) not in typing.get_args(field_type) and field not in kwargs
            ]
            if missing_keys:
                raise TypeError(
                    f"Missing required argument(s): {', '.join(missing_keys)}"
                )
            for key, expected_type in model.__annotations__.items():
                if key in kwargs and not isinstance(kwargs[key], expected_type):
                    raise ValueError(
                        f"Failed validation: Key '{key}' must be of type {expected_type}, but got {type(kwargs[key]).__name__}"
                    )
            return func(self, **kwargs)

        return wrapper

    return decorator
