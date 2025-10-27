from http import HTTPStatus

from requests import Response

from mlops_codex.base import send_http_request
from mlops_codex.utils.urls import TrainingUrl


def register(data: dict[str, str], group: str, headers: dict) -> str:
    """
    Register a new training experiment. It sends a multipart form.

    Args:
        data (dict): Text data parameters
        group (str): Group name where the experiment will be registered
        headers (dict): HTTP headers

    Returns:
        (str): Training hash
    """
    response = send_http_request(
        url=TrainingUrl.REGISTER_URL.format(group_name=group),
        method='POST',
        successful_code=HTTPStatus.CREATED,
        data=data,
        headers=headers,
    ).json()

    print(response['Message'])

    return response['TrainingHash']


def upload(
    group: str, training_hash: str, headers: dict, data: dict, files: list
) -> int:
    """
    Upload a new training experiment. It sends a multipart form.
    Args:
        group (str): Group name where the experiment will be registered
        training_hash (str): Training hash
        headers (dict): HTTP headers
        data (dict): Text data parameters
        files (list): List of files to be uploaded

    Returns:
        (int): Execution id that references to a training execution
    """
    response = send_http_request(
        url=TrainingUrl.UPLOAD_URL.format(
            group_name=group, training_hash=training_hash
        ),
        method='POST',
        successful_code=HTTPStatus.CREATED,
        data=data,
        files=files,
        headers=headers,
    ).json()

    print(response['Message'])

    return response['ExecutionId']


def execute(group: str, training_hash: str, execution_id: int, headers: dict) -> None:
    """
    Send an HTTP request to enable the training execution.

    Args:
        group (str): Group name where the experiment will be executed
        training_hash (str): Training hash
        execution_id (int): Training execution id
        headers (dict): HTTP headers
    """
    response = send_http_request(
        url=TrainingUrl.EXECUTE_URL.format(
            group_name=group, training_hash=training_hash, execution_id=execution_id
        ),
        method='GET',
        successful_code=HTTPStatus.OK,
        headers=headers,
    ).json()

    print(response['Message'])


def status(group: str, execution_id: int, headers: dict) -> Response:
    """
    Send an HTTP request to enable the training execution.

    Args:
        group (str): Group name where the experiment will be executed
        execution_id (int): Training execution id
        headers (dict): HTTP headers

    Returns:
        (Response): HTTP response
    """
    response = send_http_request(
        url=TrainingUrl.STATUS_URL.format(group_name=group, execution_id=execution_id),
        method='GET',
        successful_code=HTTPStatus.OK,
        headers=headers,
    )

    return response
