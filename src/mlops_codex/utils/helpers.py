import time
from enum import StrEnum
from typing import Callable


def wait(
    f: Callable, valid_status: list[StrEnum], status_enum, status_key: str, **kwargs
) -> dict:
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
