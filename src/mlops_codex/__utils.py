import io
import json
from functools import wraps
from typing import Tuple, Union, Type, Callable
import typing

import requests
import yaml
from cachetools.func import ttl_cache

from mlops_codex.exceptions import AuthenticationError, ServerError


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


def try_login(
    login: str, password: str, base_url: str
) -> Union[Tuple[str, str], Exception]:
    """Try to sign in MLOps

    Args:
        login: User email
        password: User password
        base_url: URL that will handle the requests

    Returns:
        User login token

    Raises:
        AuthenticationError: Raises if the `login` or `password` are wrong
        ServerError: Raises if the server is not running correctly
        BaseException: Raises if the server status is something different from 200
    """
    response = requests.get(f"{base_url}/health")

    server_status = response.status_code

    if server_status == 401:
        raise AuthenticationError("Email or password invalid.")

    if server_status >= 500:
        raise ServerError("MLOps server unavailable at the moment.")

    if server_status != 200:
        raise Exception(f"Unexpected error! {response.text}")

    token = refresh_token(login, password, base_url)
    version = response.json().get("Version")
    return token, version


@ttl_cache
def refresh_token(login: str, password: str, base_url: str):
    respose = requests.post(
        f"{base_url}/login", data={"user": login, "password": password}
    )

    if respose.status_code == 200:
        return respose.json()["Token"]
    else:
        raise AuthenticationError(respose.text)


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
                field for field, field_type in model.__annotations__.items()
                if type(None) not in typing.get_args(field_type) and field not in kwargs
            ]
            if missing_keys:
                raise TypeError(f"Missing required argument(s): {', '.join(missing_keys)}")
            for key, expected_type in model.__annotations__.items():
                if key in kwargs and not isinstance(kwargs[key], expected_type):
                    raise ValueError(
                        f"Failed validation: Key '{key}' must be of type {expected_type}, but got {type(kwargs[key]).__name__}"
                    )
            return func(self, **kwargs)
        return wrapper
    return decorator
