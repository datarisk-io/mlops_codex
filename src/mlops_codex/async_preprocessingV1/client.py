from http import HTTPStatus

from attrs import define, field
from cachetools.func import ttl_cache
from time import sleep

from pydantic import BaseModel, Field
from pathlib import Path

from mlops_codex.__model_states import ModelExecutionState
from mlops_codex.base.client import send_http_request
from mlops_codex.logger_config import get_logger
from mlops_codex.utils.urls import AsyncPreprocessingUrlV1

from models import NeomarilAsyncPreprocessingV1

logger = get_logger()


class InvalidAsyncPreprocessingError(Exception):
    """Invalid preprocessing error."""

    def __init__(
        self,
        message: str = "Not a valid Neomaril async preprocessing instance. `use_exisisting_preprocessing`.",
    ) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"Could not send request, invalid Neomaril preprocessing instance."


class AsyncPreprocessingClientV1(BaseModel):
    """
    Async preprocessing client class.

    Args:
        - bearer_token: a valid Neomaril bearer token.
    """

    bearer_token: str = Field(description="Neomaril session token")
    __neomaril_async_preprocessing = None

    def use_existing_preprocessing(self, npa: NeomarilAsyncPreprocessingV1):
        """
        Set a existing sync preprocessing.

        Args:
            - npa (NeomarilAsyncPreprocessingV1): a `search` neomaril preprocessing.
        """

        self.__neomaril_async_preprocessing = npa

    def get_status(self) -> str:
        """
        Get host status of a `async preprocessing`.
        """

        if self.__neomaril_async_preprocessing is None:
            raise InvalidAsyncPreprocessingError()

        response = send_http_request(
            url=AsyncPreprocessingUrlV1.STATUS_HOST_URL.format(
                group_name=self.__neomaril_async_preprocessing.group_name,
                script_hash=self.__neomaril_async_preprocessing.hash,
            ),
            method="GET",
            successful_code=HTTPStatus.OK,
            headers={
                "Authorization": "Bearer " + self.bearer_token,
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.get_status.__qualname__,
            },
        )

        return response.json()["Status"]

    def get_logs(self) -> str:
        """
        Get logs of a `async preprocessing`.
        """

        if self.__neomaril_async_preprocessing is None:
            raise InvalidAsyncPreprocessingError()

        response = send_http_request(
            url=AsyncPreprocessingUrlV1.LOGS_URL.format(
                group_name=self.__neomaril_async_preprocessing.group_name,
                script_hash=self.__neomaril_async_preprocessing.hash,
            ),
            method="GET",
            successful_code=HTTPStatus.OK,
            headers={
                "Authorization": "Bearer " + self.bearer_token,
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.get_status.__qualname__,
            },
        )

        return response.json()

    def describe(self) -> str:
        """
        Describe (get information) a `async preprocessing`.
        """

        if self.__neomaril_async_preprocessing is None:
            raise InvalidAsyncPreprocessingError()

        response = send_http_request(
            url=AsyncPreprocessingUrlV1.DESCRIBE_URL.format(
                group_name=self.__neomaril_async_preprocessing.group_name,
                script_hash=self.__neomaril_async_preprocessing.hash,
            ),
            method="GET",
            successful_code=HTTPStatus.OK,
            headers={
                "Authorization": "Bearer " + self.bearer_token,
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.get_status.__qualname__,
            },
        )

        return response.json()

    def host(self) -> dict[str, object]:
        """
        Start the `host` process of a `async preprocessing`.
        """

        if self.__neomaril_async_preprocessing is None:
            raise InvalidAsyncPreprocessingError()

        response = send_http_request(
            url=AsyncPreprocessingUrlV1.HOST_URL.format(
                group_name=self.__neomaril_async_preprocessing.group_name,
                script_hash=self.__neomaril_async_preprocessing.hash,
            ),
            method="GET",
            successful_code=HTTPStatus.ACCEPTED,
            headers={
                "Authorization": "Bearer " + self.bearer_token,
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.get_status.__qualname__,
            },
        )

        return response.json()

    def search(self) -> list[NeomarilAsyncPreprocessingV1]:
        """
        Search (list) for `async preprocessing`s.
        """
        logger.debug("Sending search request to neomaril.")

        response = send_http_request(
            url=f"{AsyncPreprocessingUrlV1.SEARCH_URL}?operation=Async",
            method="GET",
            successful_code=HTTPStatus.OK,
            headers={
                "Authorization": "Bearer " + self.bearer_token,
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.get_status.__qualname__,
            },
        )

        return [NeomarilAsyncPreprocessingV1(**sp) for sp in response.json()["Results"]]

    def get_execution_status(
        self, execution_id: int, group_token: str
    ) -> dict[str, object]:
        """
        Get status of `execution_id`, with the given `execution_id` and `group_token`.

        Args:
            - execution_id(int): the `execution id`.
            - group_token(str): the group token.
        """
        if self.__neomaril_async_preprocessing is None:
            raise InvalidAsyncPreprocessingError()

        response = send_http_request(
            url=AsyncPreprocessingUrlV1.STATUS_EXECUTION_URL.format(
                group_name=self.__neomaril_async_preprocessing.group_name,
                execution_id=execution_id,
            ),
            method="GET",
            successful_code=HTTPStatus.OK,
            headers={
                "Authorization": "Bearer " + group_token,
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.get_status.__qualname__,
            },
        )

        return response.json()

    def __wait_for_execution(
        self, execution_id: int, group_token: str
    ) -> dict[str, object]:
        """
        Check the execution status of the preprocessing script every 30 seconds.

        Args:
            execution_id(int): execution id of the preprocessing script.
        """
        status_response = self.get_execution_status(execution_id, group_token)
        status = status_response["Status"]

        print("Waiting for preprocessing script to finish", end="")
        while (status == ModelExecutionState.Running) or (
            status == ModelExecutionState.Requested
        ):
            sleep(30)
            print("aq")
            status_response = self.get_execution_status(execution_id, group_token)
            status = status_response["Status"]
            print(".", end="")

        if status == ModelExecutionState.Succeeded:
            logger.debug("Preprocessing script finished successfully")
            return status_response

        logger.debug(
            f"Preprocessing script execution {execution_id} is other status different than Succeeded. Current status = {status}"
        )
        return status_response

    def run_with_dataset(
        self, dataset_hash: Path, group_token: str, waitForReady: bool = False
    ) -> dict[str, object]:
        """
        Run a `async Preprocessing` with `dataset_hash`.
        """

        if self.__neomaril_async_preprocessing is None:
            raise InvalidAsyncPreprocessingError()

        data = {"dataset_hash": dataset_hash}

        logger.debug(f"sending a `run` with `dataset_hash`, {dataset_hash}")

        response = send_http_request(
            url=AsyncPreprocessingUrlV1.RUN_URL.format(
                group_name=self.__neomaril_async_preprocessing.group_name,
                script_hash=self.__neomaril_async_preprocessing.hash,
            ),
            method="POST",
            successful_code=HTTPStatus.ACCEPTED,
            headers={
                "Authorization": "Bearer " + group_token,
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.get_status.__qualname__,
            },
            data=input_str,
        )

        if waitForReady:
            execution_id = response.json()["ExecutionId"]
            return self.__wait_for_execution(execution_id, group_token)
        return response.json()

    def run_with_file(
        self, input_file_path: Path, group_token: str, waitForReady: bool = False
    ) -> dict[str, object]:
        """
        Run a `async Preprocessing` with `input_file_path`.
        """

        if self.__neomaril_async_preprocessing is None:
            raise InvalidAsyncPreprocessingError()

        if input_file_path.exists():
            files = {"dataset": open(input_file_path, "rb")}

            logger.debug("found dataset file, sending request to neomaril.")

            response = send_http_request(
                url=AsyncPreprocessingUrlV1.RUN_URL.format(
                    group_name=self.__neomaril_async_preprocessing.group_name,
                    script_hash=self.__neomaril_async_preprocessing.hash,
                ),
                method="POST",
                successful_code=HTTPStatus.ACCEPTED,
                headers={
                    "Authorization": "Bearer " + group_token,
                    "Neomaril-Origin": "Codex",
                    "Neomaril-Method": self.get_status.__qualname__,
                },
                files=files,
            )

            if waitForReady:
                execution_id = response.json()["ExecutionId"]
                return self.__wait_for_execution(execution_id, group_token)
            return response.json()

        else:
            logger.debug(f"file: {input_file_path} doesn't exists")
            return None
