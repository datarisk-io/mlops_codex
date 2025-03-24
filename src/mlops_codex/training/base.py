import abc
from time import sleep
from typing import Any

from pydantic import BaseModel, Field

from mlops_codex.__model_states import ModelExecutionState
from mlops_codex.base import BaseMLOps
from mlops_codex.exceptions import TrainingError
from mlops_codex.http_request_handler import make_request, refresh_token
from mlops_codex.logger_config import get_logger

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
        raise NotImplementedError("Promotion is not implemented.")

    def execution_info(self):
        """
        Abstract method to get execution information.

        Returns
        -------
        None
        """
        raise NotImplementedError("Execution info is not implemented.")

    def copy_execution(self):
        url = f"{self.mlops_class.base_url}/v2/training/execution/{self.execution_id}/copy"
        token = refresh_token(*self.mlops_class.credentials, self.mlops_class.base_url)
        return self._do_copy(
            url, token, self.group, self.experiment_name, self.mlops_class

        )

    @classmethod
    @abc.abstractmethod
    def _do_copy(cls, url, token, group, experiment_name, mlops_class):
        pass