from typing import Annotated

from pydantic import BaseModel, BeforeValidator, FilePath, PositiveInt

from mlops_codex.administrator.auth import AuthManager
from mlops_codex.train.client import status
from mlops_codex.utils.services_status import ExecutionStatus
from mlops_codex.utils.validations import str_to_path
from mlops_codex.utils.validations import validate_python_version


class MLOpsExperiment(BaseModel):
    """
    A MLOps experiment

    Args:
        training_hash (str): The hash of the training run
        experiment_name (str): The name of the experiment
    """

    training_hash: str
    experiment_name: str
    model_type: str
    group: str


class MLOpsTrainExecution(BaseModel):
    """
    A MLOps training execution
    """

    experiment: MLOpsExperiment
    execution_id: PositiveInt
    auth: AuthManager

    @property
    def status(self) -> str:
        response = status(
            group=self.experiment.group,
            execution_id=self.execution_id,
            headers=self.auth.header,
        ).json()
        return ExecutionStatus(response['Status'])


class CustomTrain(BaseModel):
    training_reference: str
    run_name: str
    python_version: Annotated[str, BeforeValidator(validate_python_version)]
    input_data: str
    source: Annotated[str | FilePath, BeforeValidator(str_to_path)]
    requirements: Annotated[str | FilePath, BeforeValidator(str_to_path)]
    env_file: Annotated[str | FilePath | None, BeforeValidator(str_to_path)] = None
    extras: list[Annotated[str | FilePath | None, BeforeValidator(str_to_path)]] = None

    def __repr__(self):
        return 'Custom'


class AutoMLTrain(BaseModel):
    run_name: str
    input_data: str
    configuration: Annotated[str, BeforeValidator(str_to_path)]

    def __repr__(self):
        return 'AutoML'
