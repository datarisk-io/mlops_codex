from http import HTTPStatus
from typing import Dict, Union

import requests


class APIException(Exception):
    """Generic API exception"""

    def __init__(self, response: requests.Response, status_code: int):
        self.response = response
        self.status_code = status_code

    def __str__(self) -> str:
        return (
            f'Error duo to request to MLOps API. Status code: {self.status_code}'
            f' Response: {self.response.text}'
        )


class InsufficientPermissions(Exception):
    """Raised when a user does not have enough permissions"""

    def __init__(
        self,
        response: requests.Response,
        message=(
            'The provided token does not have enough permissions. '
            'Check your credentials or group token'
        ),
    ):
        super().__init__(message)
        self.message = message
        self.status_code = HTTPStatus.UNAUTHORIZED
        self.response = response

    def __str__(self):
        return f'{self.message} \n Status code: 401|> Response: {self.response.text}'


class BadRequest(Exception):
    """Raised when the request is a bad request"""

    def __init__(
        self,
        response: requests.Response,
        message=(
            'The request was unsuccessful due to a bad request. '
            'Maybe the request syntax is wrong.'
        ),
    ):
        super().__init__(message)
        self.message = message
        self.status_code = HTTPStatus.BAD_REQUEST
        self.response = response

    def __str__(self):
        return f'{self.message} \n Status code: 400|> Response: {self.response.text}'


class NotFound(Exception):
    """Raised when the requested resource is not found"""

    def __init__(self, response: requests.Response, message='Resource was not found.'):
        super().__init__(message)
        self.message = message
        self.status_code = HTTPStatus.NOT_FOUND
        self.response = response

    def __str__(self):
        return f'{self.message} \n Status code: 404|> Response: {self.response.text}'


class APIConnectionError(Exception):
    """The request was unsuccessful due to a connection error"""

    def __init__(
        self,
        response: requests.Response,
        message: str = (
            'The request was unsuccessful due to a connection error. '
            'Check your internet connection'
        ),
    ) -> None:
        super().__init__(message)
        self.message = message
        self.response = response

    def __str__(self):
        return (
            f'{self.message} \n Status code: {self.response.status_code}'
            f'|> Response: {self.response.text}'
        )


class APITimeoutError(Exception):
    """The request got timed out. You might try checking"""

    def __init__(
        self,
        response: requests.Response,
        message: str = 'Request timed out. Check your internet connection',
    ) -> None:
        super().__init__(message)
        self.message = message
        self.response = response

    def __str__(self):
        return (
            f'{self.message} \n Status code: {self.response.status_code}'
            f'|> Response: {self.response.text}'
        )


def raise_for_status(response: requests.Response, successful_code: HTTPStatus) -> None:
    """
    Raise an exception based on the response status code if the status code is not equal
    to the successful status code.

    Args:
        response (requests.Response): The response from the request.
        successful_code (HTTPStatus): The status code of the successful response.

    Raises:
        BadRequest: The request was unsuccessful due to a bad request.
        APIConnectionError: The request was unsuccessful due to a connection error.
        InsufficientPermissions: If the user does not have enough permissions.
        NotFound: The request was unsuccessful due to a resource not found.
    """
    code_exc_dict: Dict[Union[int, HTTPStatus], Exception] = {
        HTTPStatus.UNAUTHORIZED: InsufficientPermissions(response=response),
        HTTPStatus.BAD_REQUEST: BadRequest(response=response),
        HTTPStatus.NOT_FOUND: NotFound(response=response),
    }

    status_code = response.status_code
    if status_code == successful_code:
        return

    if status_code not in code_exc_dict:
        raise APIException(response=response, status_code=status_code)

    raise code_exc_dict.get(
        response.status_code,
        APIException(response=response, status_code=status_code),
    )
