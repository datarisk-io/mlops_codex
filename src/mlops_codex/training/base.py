import abc
from time import sleep
from typing import Any

from pydantic import BaseModel, Field

from mlops_codex.__model_states import ModelExecutionState
from mlops_codex.base import BaseMLOps
from mlops_codex.exceptions import InputError, TrainingError
from mlops_codex.http_request_handler import make_request, refresh_token
from mlops_codex.logger_config import get_logger
from mlops_codex.model import AsyncModel, SyncModel

logger = get_logger()


class ITrainingExecution(BaseModel, abc.ABC):
    """
    Interface for training execution.

    Parameters
    ----------
    training_hash: str
        Training hash.
    group: str
        Group where the training is inserted.
    model_type: str
        Type of the model. It must be 'Custom', 'AutoML' or 'External'
    execution_id: int
        Execution ID of a training.
    experiment_name: str
        Name of the experiment.
    login: str
        Login credential.
    password: str
        Password credential.
    url: str
        Url used to connect to the MLOps server.
    mlops_class: BaseMLOps
        MLOps class instance.
    """

    training_hash: str = Field(
        frozen=True, title="Training hash", validate_default=True
    )
    group: str = Field(frozen=True, title="Training hash", validate_default=True)
    model_type: str = Field(frozen=True, title="Training type", validate_default=True)

    execution_id: int = Field(default=None, gt=0)
    experiment_name: str = Field(default=None)

    login: str = Field(default=None, repr=False)
    password: str = Field(default=None, repr=False)
    url: str = Field(default="https://neomaril.datarisk.net/", repr=False)
    mlops_class: BaseMLOps = Field(default=None, repr=False)

    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes the model after creation.

        Parameters
        ----------
        __context: Any
            Context for initialization.
        """
        if self.mlops_class is None:
            self.mlops_class = BaseMLOps(
                login=self.login, password=self.password, url=self.url
            )

        url = f"{self.mlops_class.base_url}/training/describe/{self.group}/{self.training_hash}"
        token = refresh_token(*self.mlops_class.credentials, self.mlops_class.base_url)

        response = make_request(
            url=url,
            method="GET",
            success_code=200,
            custom_exception=TrainingError,
            custom_exception_message=f'Experiment "{self.training_hash}" not found.',
            specific_error_code=404,
            logger_msg=f'Experiment "{self.training_hash}" not found.',
            headers={
                "Authorization": f"Bearer {token}",
            },
        )

        training_data = response.json()["Description"]
        self.experiment_name = training_data["ExperimentName"]

    def _register_execution(self, run_name: str, description: str, training_type: str):
        """
        Registers a new execution.

        Parameters
        ----------
        run_name: str
            Name of the run.
        description: str
            Description of the run.
        training_type: str
            Type of the training.

        Returns
        -------
        int
            Execution ID.
        """
        url = f"{self.mlops_class.base_url}/v2/training/{self.training_hash}/execution"
        token = refresh_token(*self.mlops_class.credentials, self.mlops_class.base_url)
        payload = {
            "RunName": run_name,
            "Description": description,
            "TrainingType": training_type,
        }
        response = make_request(
            url=url,
            method="POST",
            success_code=201,
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )

        msg = response.json().get("Message")
        execution_id = response.json()["ExecutionId"]
        logger.info(f"{msg} for {run_name}")
        return execution_id

    def _upload_input_file(self, input_data: str, upload_data: str):
        """
        Uploads an input file.

        Parameters
        ----------
        input_data: str
            Input data to be uploaded. It is the payload.
        upload_data: str
            Data for the upload. It is the file.

        Returns
        -------
        None
        """
        url = f"{self.mlops_class.base_url}/v2/training/execution/{self.execution_id}/input/file"
        token = refresh_token(*self.mlops_class.credentials, self.mlops_class.base_url)

        response = make_request(
            url=url,
            method="PATCH",
            success_code=201,
            data=input_data,
            files=upload_data,
            headers={
                "Authorization": f"Bearer {token}",
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self._upload_input_file.__qualname__,
            },
        )

        msg = response.json()["Message"]
        dataset_hash = response.json()["DatasetHash"]

        logger.info(f"{msg}")
        logger.info(f"Dataset hash = {dataset_hash}")

    def _upload_requirements(self, requirements_file: str):
        """
        Uploads a requirements file.

        Parameters
        ----------
        requirements_file: str
            Path to the requirements file.

        Returns
        -------
        None
        """
        url = f"{self.mlops_class.base_url}/v2/training/execution/{self.execution_id}/requirements-file"
        token = refresh_token(*self.mlops_class.credentials, self.mlops_class.base_url)

        upload_data = {"requirements": open(requirements_file, "rb")}

        response = make_request(
            url=url,
            method="PATCH",
            success_code=201,
            files=upload_data,
            headers={
                "Authorization": f"Bearer {token}",
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self._upload_requirements.__qualname__,
            },
        )

        msg = response.json()["Message"]
        logger.info(msg)

    def _upload_extra_files(self, extra_files_path: str, name: str):
        """
        Uploads extra file.

        Parameters
        ----------
        extra_files_path: str
            Path to the extra file.
        name: str
            Name of the file.

        Returns
        -------
        None
        """
        url = f"{self.mlops_class.base_url}/v2/training/execution/{self.execution_id}/extra-files"
        token = refresh_token(*self.mlops_class.credentials, self.mlops_class.base_url)

        upload_data = {"extra": open(extra_files_path, "rb")}
        input_data = {"file_name": name}
        response = make_request(
            url=url,
            method="PATCH",
            success_code=201,
            data=input_data,
            files=upload_data,
            headers={
                "Authorization": f"Bearer {token}",
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self._upload_extra_files.__qualname__,
            },
        )

        msg = response.json()["Message"]
        logger.info(msg)

    def _upload_env_file(self, env_file: str):
        """
        Uploads an environment file.

        Parameters
        ----------
        env_file: str
            Path to the environment file.

        Returns
        -------
        None
        """
        url = f"{self.mlops_class.base_url}/v2/training/execution/{self.execution_id}/env/file"
        token = refresh_token(*self.mlops_class.credentials, self.mlops_class.base_url)

        upload_data = {"env": open(env_file, "rb")}
        response = make_request(
            url=url,
            method="PATCH",
            success_code=201,
            files=upload_data,
            headers={
                "Authorization": f"Bearer {token}",
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self._upload_extra_files.__qualname__,
            },
        )

        msg = response.json()["Message"]
        logger.info(msg)

    @property
    def status(self) -> str:
        """
        Gets the current status of the execution.

        Returns
        -------
        str
            Current status of the execution.

        Raises
        ------
        TrainingError
            If the execution is not found.
        AuthenticationError
            If the authentication fails.
        """
        url = f"{self.mlops_class.base_url}/v2/training/execution/{self.execution_id}/status"
        token = refresh_token(*self.mlops_class.credentials, self.mlops_class.base_url)
        response = make_request(
            url=url,
            method="GET",
            success_code=200,
            custom_exception=TrainingError,
            custom_exception_message=f"Experiment with execution id {self.execution_id} not found.",
            specific_error_code=404,
            logger_msg=f"Experiment with execution id {self.execution_id} not found.",
            headers={
                "Authorization": f"Bearer {token}",
            },
        ).json()

        status = response["Status"]
        if status == "Failed":
            msg = response["Message"]
            logger.info(msg)

        return status

    def host(self):
        """
        Hosts the current execution.

        Returns
        -------
        None
        """
        url = f"{self.mlops_class.base_url}/v2/training/execution/{self.execution_id}"
        token = refresh_token(*self.mlops_class.credentials, self.mlops_class.base_url)

        response = make_request(
            url=url,
            method="PATCH",
            success_code=202,
            headers={
                "Authorization": f"Bearer {token}",
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.host.__qualname__,
            },
        ).json()

        msg = response["Message"]
        logger.info(msg)

    def wait_ready(self):
        """
        Waits until the model is ready.

        Returns
        -------
        None
        """
        current_status = ModelExecutionState.Running
        print("Training your model...", end="", flush=True)
        while current_status in [
            ModelExecutionState.Running,
            ModelExecutionState.Requested,
        ]:
            current_status = ModelExecutionState[self.status]
            sleep(30)
            print(".", end="", flush=True)
        print()

        if current_status == ModelExecutionState.Succeeded:
            logger.info("Training finished successfully.")
        else:
            logger.info(f"Training failed. Current status is {current_status}")

    # TODO: update it when promote endpoint updated
    def __promote_validation(self, **kwargs):
        """
        Validates the promotion process.

        Parameters
        ----------
        kwargs: dict
            Keyword arguments for validation.

        Returns
        -------
        None
        """
        if kwargs["operation"] == "Async":
            if kwargs["input_type"] is None:
                raise InputError(
                    "For asynchronous models, you must provide the 'input_type' argument."
                )
            if kwargs["schema_extension"]:
                raise InputError(
                    "The input schema extension must be json, csv, parquet or xml."
                )

        status = self.status
        if status != "Succeeded":
            raise TrainingError(
                f"Training execution must be Succeeded to be promoted, current status is {status}"
            )

    def __promote(self, **kwargs):
        """
        Promotes the current execution.

        Parameters
        ----------
        kwargs: dict
            Keyword arguments for promotion.

        Returns
        -------
        Union[SyncModel, AsyncModel]
            Promoted model instance.
        """
        url = f"{self.mlops_class.base_url}/training/promote/{self.group}/{self.training_hash}/{self.execution_id}"
        token = refresh_token(*self.mlops_class.credentials, self.mlops_class.base_url)

        upload_data = kwargs.get("upload_data")
        input_data = kwargs.get("input_data")

        response = make_request(
            url=url,
            method="POST",
            success_code=201,
            data=input_data,
            files=upload_data,
            headers={
                "Authorization": f"Bearer {token}",
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.__promote.__qualname__,
            },
        )

        msg = response.json()["Message"]
        logger.info(msg)

        model_hash = response["ModelHash"]
        operation = kwargs["operation"]
        model_name = kwargs["model_name"]

        builder = SyncModel if operation.title() == "Sync" else AsyncModel
        model = builder(
            name=model_name,
            model_hash=model_hash,
            login=self.mlops_class.credentials[0],
            password=self.mlops_class.credentials[1],
            url=self.mlops_class.base_url,
            group=self.group,
        )

        model.host(operation=operation.title())

        wait_complete = kwargs.get("wait_complete", False)
        if wait_complete:
            model.wait_ready()

        return model

    @abc.abstractmethod
    def promote(self, *args, **kwargs):
        """
        Abstract method to promote the execution.

        Parameters
        ----------
        args: tuple
            Positional arguments.
        kwargs: dict
            Keyword arguments.

        Returns
        -------
        None
        """
        raise NotImplementedError()

    def execution_info(self):
        """
        Abstract method to get execution information.

        Returns
        -------
        None
        """
        raise NotImplementedError()
