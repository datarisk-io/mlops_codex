from http import HTTPStatus
from typing import Literal

import requests

from V2.utils.exceptions import raise_for_status, APITimeoutError, APIConnectionError


def send_http_request(
        url: str,
        method: Literal['GET', 'POST', 'PUT', 'PATCH', 'DELETE'],
        successful_code: HTTPStatus,
        **kwargs
) -> requests.Response:
    """
    Generic function used to make a request to MLOps server.

    Args:
        url (str): URL of the endpoint.
        method (str): HTTP method (get, post, put, patch and delete).
        successful_code (HTTPStatus): Success HTTP status code returned by the API

    Returns:
        requests.Response

    Raises:
        requests.exceptions.RequestException: If the request fails.
    """
    request = requests.Request(
        method=method,
        url=url,
        **kwargs
    )

    try:

        prepared_request = request.prepare()
        with requests.Session() as session:
            response = session.send(prepared_request, timeout=60)

        raise_for_status(response, successful_code)

        return response

    except requests.exceptions.Timeout:
        raise APITimeoutError(response)

    except requests.exceptions.ConnectionError:
        raise APIConnectionError(response)
