from typing import Tuple, Union

import requests
from cachetools.func import ttl_cache

from mlops_codex.__utils import parse_json_to_yaml
from mlops_codex.exceptions import (
    AuthenticationError,
    InputError,
    ServerError,
    UnexpectedError,
)
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
        ServerError: Raises if the server is not correctly running
        BaseException: Raises if the server status is different from 200
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
    response = requests.post(
        f"{base_url}/login",
        data={"user": login, "password": password},
        timeout=60,
    )

    if response.status_code == 200:
        return response.json()["Token"]
    else:
        raise AuthenticationError(response.text)


def handle_common_errors(
    response: requests.Response,
    specific_error_code,
    custom_exception,
    custom_exception_message,
    logger_msg,
):
    """
    Handle possible errors

    Args:
        response (requests.Response): Response from MLOps server
        specific_error_code (int): Error code
        custom_exception (_SpecialForm[Exception]): Custom exception
        custom_exception_message (str): Custom exception message
        logger_msg (str): Log message
    """
    if response.status_code == 401:
        raise AuthenticationError("Unauthorized: Check your credentials or token.")
    elif response.status_code == 400:
        logger.error(parse_json_to_yaml(response.json()))
        raise InputError("The request had a error in the input.")
    elif response.status_code >= 500:
        raise ServerError("Server is down or unavailable.")
    elif specific_error_code == response.status_code:
        if logger_msg:
            logger.info(logger_msg)
        else:
            logger.info(response.json())
        raise custom_exception(custom_exception_message)

    formatted_msg = parse_json_to_yaml(response.json())
    logger.info(f"Something went wrong. \n{formatted_msg}")
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
    files=None,
    timeout=60,
):
    """
    Makes a generic HTTP request.

    Args:
        url (str): URL of the endpoint.
        method (str): HTTP method (get, post, delete, patch, etc).
        success_code (int): Status codes indicating success.
        custom_exception (_SpecialForm[Exception]): Custom exception class.
        custom_exception_message (str): Custom exception message.
        specific_error_code (int): Specific error code.
        logger_msg (str): Logger message.
        headers (dict, optional): Request headers.
        params (dict, optional): URL parameters for GET requests.
        data (dict, optional): Data for POST/PUT requests (form-encoded).
        json (dict, optional): Data for POST/PUT requests (JSON).
        files (dict, optional): Data for POST/PUT requests (files).
        timeout (int, optional): Timeout in seconds for the request. Default is 60.

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
        files=files,
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
