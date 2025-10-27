from http import HTTPStatus

from requests import Response

from mlops_codex.base import send_http_request
from mlops_codex.utils.urls import TrainingUrl


def register(data: dict[str, str], group: str, headers: dict) -> str:
    response = send_http_request(
        url=TrainingUrl.REGISTER_URL.format(group_name=group),
        method='POST',
        successful_code=HTTPStatus.CREATED,
        data=data,
        headers=headers,
    ).json()

    print(response['Message'])

    return response['TrainingHash']


def upload(group: str, training_hash: str, headers: dict, data: dict, files: list) -> int:
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
    response = send_http_request(
        url=TrainingUrl.STATUS_URL.format(group_name=group, execution_id=execution_id),
        method='GET',
        successful_code=HTTPStatus.OK,
        headers=headers,
    )

    return response
