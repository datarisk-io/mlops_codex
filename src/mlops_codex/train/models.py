from typing import Annotated

from pydantic import AfterValidator, BaseModel, BeforeValidator, FilePath, PositiveInt

from mlops_codex.administrator.auth import AuthManager
from mlops_codex.train.client import status
from mlops_codex.utils.services_status import ExecutionStatus
from mlops_codex.utils.validations import (
    file_extension_validation,
    str_to_path,
    validate_python_version,
)


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

    Args:
        experiment (MLOpsExperiment): The experiment object
        execution_id (str): The execution id of the training execution
        auth (AuthManager): Auth manager.
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
    """
    A custom training run

    Args:
        training_reference (str): Entrypoint function name
        run_name (str): Experiment name
        python_version (str): Python version. Available versions are 3.8, 3.9 and 3.10
        input_data (str): Input data. It can be a path to a file or a dataset hash which a string
        source (str | FilePath): Path to the .py script with an entry point function
        requirements (str | FilePath): Path to the requirements file. It must be a txt file
        env_file (str | FilePath | None): Path to the .env file
        extras (list[str | FilePath | None]): Paths to extras file. It can be a list of extra files

    """
    training_reference: str
    run_name: str
    python_version: Annotated[str, BeforeValidator(validate_python_version)]
    input_data: str
    source: Annotated[
        str | FilePath,
        BeforeValidator(str_to_path),
        AfterValidator(file_extension_validation('.py')),
    ]
    requirements: Annotated[
        str | FilePath,
        BeforeValidator(str_to_path),
        AfterValidator(file_extension_validation('.txt')),
    ]
    env_file: Annotated[
        str | FilePath | None,
        BeforeValidator(str_to_path),
        AfterValidator(file_extension_validation('.env')),
    ] = None
    extras: list[Annotated[str | FilePath | None, BeforeValidator(str_to_path)]] = None

    def __repr__(self):
        return 'Custom'


class AutoMLTrain(BaseModel):
    """
    A AutoML training run

    Args:
        run_name (str): Experiment name
        input_data (str): Input data. It can be a path to a file or a dataset hash which a string
        configuration (str | FilePath): Path to the configuration file. It must be a json file
    """
    run_name: str
    input_data: str
    configuration: Annotated[
        str,
        BeforeValidator(str_to_path),
        AfterValidator(file_extension_validation('.json')),
    ]

    def __repr__(self):
        return 'AutoML'


class External(BaseModel):
    """
    An external training run

    Args:
        run_name (str): Experiment name
        python_version (str): Python version. Available versions are 3.8, 3.9 and 3.10
        features (pathlib.Path): Input features used to train the model, needs to be a .parquet
        target (pathlib.Path): A .parquet file with the targets used to train
        output (pathlib.Path): A .parquet file with the predictions returned from the trained model
        metrics (pathlib.Path | None): A .json file with training metrics of the trained model
        model (pathlib.Path | None): A binary file with the model trained to be executed at the API. Allowed extensions are: .pkl, .pickle, .cbm, .json, .txt and .h5
        requirements (pathlib.Path | None): A .txt file with the packages used in the model
        parameters (pathlib.Path | None): A .json file containing experiment parameters
        model_hash (str | None): .json file containing experiment parameters

    """
    run_name: str
    python_version: Annotated[str, BeforeValidator(validate_python_version)]
    features: Annotated[
        str | FilePath,
        BeforeValidator(str_to_path),
        AfterValidator(file_extension_validation('.parquet')),
    ]
    target: Annotated[
        str | FilePath,
        BeforeValidator(str_to_path),
        AfterValidator(file_extension_validation('.parquet')),
    ]
    output: Annotated[
        str | FilePath,
        BeforeValidator(str_to_path),
        AfterValidator(file_extension_validation('.parquet')),
    ]
    metrics: Annotated[
        str | FilePath | None,
        BeforeValidator(str_to_path),
        AfterValidator(file_extension_validation('.json')),
    ] = None
    model: Annotated[
        str | FilePath | None,
        BeforeValidator(str_to_path),
        AfterValidator(
            file_extension_validation('.pkl', '.pickle', '.cbm', '.json', '.txt', '.h5')
        ),
    ] = None
    requirements: Annotated[
        str | FilePath | None,
        BeforeValidator(str_to_path),
        AfterValidator(file_extension_validation('.txt')),
    ] = None
    parameters: Annotated[
        str | FilePath | None,
        BeforeValidator(str_to_path),
        AfterValidator(file_extension_validation('.json')),
    ] = None
    model_hash: str | None = None

    def __repr__(self):
        return 'External'
