from http import HTTPStatus

from attrs import define, field
from cachetools.func import ttl_cache

from pydantic import BaseModel, Field

from mlops_codex.base.client import send_http_request
from mlops_codex.logger_config import get_logger
from mlops_codex.utils.urls import AdminUrl, SyncPreprocessingUrl

from models import SyncPreprocessing, NeomarilSyncPreprocessing

logger = get_logger()


class InvalidSyncPreprocessingError(Exception):
    """Invalid sync preprocessing error."""

    def __init__(
        self,
        message: str = """
            Not a valid Neomaril sync preprocessing instance.
            `register` or `use_exisisting_preprocessing` to use the current function.",
        """,
    ) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"Could not send request, invalid Neomaril sync preprocessing instance."


class SyncPreprocessingClient(BaseModel):
    """
    Client to handle all `SyncPreprocessing` Neomaril operations.

    Args:
        - bearer_token: a valid Neomaril bearer token.
    """

    bearer_token: str = Field(description="Neomaril session token")
    __neomaril_sync_preprocessing = None

    def register(self, sp: SyncPreprocessing) -> NeomarilSyncPreprocessing:
        """
        Register a new `syncPreprocessing` in Neomaril.

        Args:
            - sp(SyncPreprocessing): a valid `SyncPreprocessing`.

        Returns:
            NeomarilSyncPreprocessing: a `created` Neomaril sync preprocessing entity.
        """
        logger.debug("Opening files to send.")

        files = {
            "source": open(sp.source_file_path, "rb"),
            "requirements": open(sp.requirements_file_path, "rb"),
            "schema": open(sp.schema_file_path, "rb"),
        }

        if sp.env_file_path is not None:
            files["env"] = open(sp.env_file_path, "rb")

        if sp.extra_file_paths is not None:
            for extra_file in sp.extra_file_paths:
                files["extra"] = open(extra_file, "rb")

        data = {
            "script_reference": sp.script_reference,
            "name": sp.name,
            "python_version": sp.python_version,
            "operation": "Sync",
        }

        logger.debug("Sending neomaril request to create sync preproc.")

        response = send_http_request(
            url=SyncPreprocessingUrl.REGISTER_URL.format(group_name=sp.group_name),
            method="POST",
            successful_code=HTTPStatus.CREATED,
            headers={
                "Authorization": "Bearer " + self.bearer_token,
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.register.__qualname__,
            },
            data=data,
            files=files,
        )

        logger.debug(f"Neomaril response, {response}.")

        nsp_json = response.json()
        nsp_json["Group"] = sp.group_name
        nsp_json["Status"] = "Ready"
        nsp_json["PythonVersion"] = sp.python_version

        nsp = NeomarilSyncPreprocessing(**nsp_json)

        self.__neomaril_sync_preprocessing = nsp

        return nsp

    def use_existing_preprocessing(self, nps: NeomarilSyncPreprocessing):
        """
        Set a existing sync preprocessing.

        Args:
            - nps(NeomarilSyncPreprocessing): a `search` neomaril preprocessing.
        """

        self.__neomaril_sync_preprocessing = nps

    def get_status(self) -> str:
        """
        Get host status of a `sync preprocessing`.
        """

        if self.__neomaril_sync_preprocessing is None:
            raise InvalidSyncPreprocessingError()

        response = send_http_request(
            url=SyncPreprocessingUrl.STATUS_HOST_URL.format(
                group_name=self.__neomaril_sync_preprocessing.group_name,
                script_hash=self.__neomaril_sync_preprocessing.hash,
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
        Get logs of a `sync preprocessing`.
        """

        if self.__neomaril_sync_preprocessing is None:
            raise InvalidSyncPreprocessingError()

        response = send_http_request(
            url=SyncPreprocessingUrl.LOGS_URL.format(
                group_name=self.__neomaril_sync_preprocessing.group_name,
                script_hash=self.__neomaril_sync_preprocessing.hash,
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
        Describe (get information) of a `sync preprocessing`.
        """

        if self.__neomaril_sync_preprocessing is None:
            raise InvalidSyncPreprocessingError()

        response = send_http_request(
            url=SyncPreprocessingUrl.DESCRIBE_URL.format(
                group_name=self.__neomaril_sync_preprocessing.group_name,
                script_hash=self.__neomaril_sync_preprocessing.hash,
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
        Start the `host` process of a `sync preprocessing`.
        """

        if self.__neomaril_sync_preprocessing is None:
            raise InvalidSyncPreprocessingError()

        response = send_http_request(
            url=SyncPreprocessingUrl.HOST_URL.format(
                group_name=self.__neomaril_sync_preprocessing.group_name,
                script_hash=self.__neomaril_sync_preprocessing.hash,
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

    def search(self) -> list[NeomarilSyncPreprocessing]:
        """
        Search (list) for Sync preprocessings.
        """
        response = send_http_request(
            url=f"{SyncPreprocessingUrl.SEARCH_URL}?operation=Sync",
            method="GET",
            successful_code=HTTPStatus.OK,
            headers={
                "Authorization": "Bearer " + self.bearer_token,
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.get_status.__qualname__,
            },
        )

        return [NeomarilSyncPreprocessing(**sp) for sp in response.json()["Results"]]

    def run(self, input_str: str, group_token: str) -> dict[str, object]:
        """
        Run a `sync Preprocessing` with `input_json` and `group_token`.
        Args:
            - input_str(str): the input of the `preprocessing` to run.
            - group_token(str): the group token.
        """

        if self.__neomaril_sync_preprocessing is None:
            raise InvalidSyncPreprocessingError()

        response = send_http_request(
            url=SyncPreprocessingUrl.RUN_URL.format(
                group_name=self.__neomaril_sync_preprocessing.group_name,
                script_hash=self.__neomaril_sync_preprocessing.hash,
            ),
            method="POST",
            successful_code=HTTPStatus.OK,
            headers={
                "Authorization": "Bearer " + group_token,
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.get_status.__qualname__,
            },
            data=input_str,
        )

        return response.json()
