from typing import Tuple, Type, Union

import requests
from cachetools.func import ttl_cache

from mlops_codex.__utils import parse_json_to_yaml
from mlops_codex.exceptions import AuthenticationError, ServerError, UnexpectedError
from mlops_codex.logger_config import get_logger

logger = get_logger()


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
    response = requests.get(f"{base_url}/health", timeout=60)

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
        f"{base_url}/login", data={"user": login, "password": password},
        timeout=60,
    )

    if respose.status_code == 200:
        return respose.json()["Token"]
    else:
        raise AuthenticationError(respose.text)


def handle_common_errors(
    response: requests.Response,
    specific_error_code: int,
    custom_exception: Type[Exception],
    custom_exception_message: str,
    logger_msg: str,
):
    """
    Handle possible errors

    Args:
        response (requests.Response): Response from MLOps server
        specific_error_code (int): Error code
        custom_exception (Type[Exception]): Custom exception
        custom_exception_message (str): Custom exception message
        logger_msg (str): Log message
    """
    if response.status_code == 401:
        raise AuthenticationError("Unauthorized: Check your credentials or token.")
    elif response.status_code >= 500:
        raise ServerError("Server is down or unavailable.")
    elif specific_error_code == response.status_code:
        if logger_msg:
            logger.error(logger_msg)
        else:
            logger.error(response.json())
        raise custom_exception(custom_exception_message)

    formatted_msg = parse_json_to_yaml(response.json())
    logger.error(f"Something went wrong. \n{formatted_msg}")
    raise UnexpectedError(
        "Unexpected error during HTTP request. Please contact the administrator."
    )


def make_request(
    url: str,
    method: str,
    success_code: int,
    custom_exception=None,
    custom_exception_message=None,
    specific_error_code=None,
    logger_msg=None,
    headers=None,
    params=None,
    data=None,
    json=None,
    timeout=60,
) -> requests.Response:
    """
    Makes a generic HTTP request.

    Args:
        url (str): URL of the endpoint.
        method (str): HTTP method (get, post, delete, patch, etc).
        success_code (int): Status codes indicating success.
        custom_exception (Type[Exception]): Custom exception class.
        custom_exception_message (str): Custom exception message.
        specific_error_code (int): Specific error code.
        logger_msg (str): Logger message.
        headers (dict, optional): Request headers.
        params (dict, optional): URL parameters for GET requests.
        data (dict, optional): Data for POST/PUT requests (form-encoded).
        json (dict, optional): Data for POST/PUT requests (JSON).
        timeout (int, optional): Timeout in seconds for the request. Default is 10.

    Returns:
        requests.Response

    Raises:
        requests.exceptions.RequestException: If the request fails.
    """
    response = requests.request(
        method=method,
        url=url,
        headers=headers,
        params=params,
        data=data,
        json=json,
        timeout=timeout,
    )

    if response.status_code == success_code:
        return response
    handle_common_errors(
        response,
        specific_error_code,
        custom_exception,
        custom_exception_message,
        logger_msg,
    )