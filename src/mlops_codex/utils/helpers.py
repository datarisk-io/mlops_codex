import time
from enum import StrEnum
from typing import Callable


def wait(
    f: Callable, valid_status: list[StrEnum], status_enum, status_key: str, **kwargs
) -> dict:
    """
    Generic function to wait until an execution achieve a valid status.

    Args:
        f (Callable): Function to be executed
        valid_status (list[StrEnum]): List of valid statuses. Not only ok, but bad status as well
        status_enum: List of status enums
        status_key (str): Key to check the endpoint
        **kwargs:
            Args passed to f
    Returns:
        (dict): Dictionary of the f results
    """
    print('Waiting for ready', end='')
    current_status = None
    json_response = {}
    while current_status not in valid_status:
        time.sleep(30)
        json_response = f(**kwargs).json()
        print('.', end='', flush=True)
        current_status = status_enum(json_response[status_key])
    print()
    return json_response
