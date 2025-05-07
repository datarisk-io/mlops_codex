from typing import Union

from pydantic import model_validator

from mlops_codex.__utils import parse_dict_or_file
from mlops_codex.dataset import MLOpsDataset, validate_dataset
from mlops_codex.exceptions import InputError
from mlops_codex.http_request_handler import make_request, refresh_token
from mlops_codex.logger_config import get_logger
from mlops_codex.shared.utils import parse_data
from mlops_codex.trainingv2.base import ITrainingExecution
from mlops_codex.trainingv2.trigger import (
    trigger_automl_training,
    trigger_custom_training,
    trigger_external_training,
)
from mlops_codex.trainingv2.validations import validate_input
from mlops_codex.validations import file_extension_validation

logger = get_logger()


class CustomTrainingExecution(ITrainingExecution):
    """
    Custom training execution class.

    Parameters
    ----------
    training_hash: str
        Training hash.
    group: str
        Group where the training is inserted.
    model_type: str
        Type of the model.
    execution_id: int
        Execution ID of a training.
    experiment_name: str
        Name of the experiment.
    login: str
        Login credential.
    password: str
        Password credential.
    url: str
        URL used to connect to the MLOps server.
    mlops_class: BaseMLOps
        MLOps class instance.
    """

    @model_validator(mode="before")
    @classmethod
    def validate(cls, values):
        """
        Validates the input values for custom training execution.

        Parameters
        ----------
        values: dict
            Dictionary of input values.

        Returns
        -------
        dict
            Validated input values.
        """

        logger.info("Validating data...")

        fields_required = (
            "input_data",
            "upload_data",
            "run_name",
            "source_file",
            "requirements_file",
            "training_reference",
            "python_version",
        )

        validate_input(fields_required, values)

        source_file = values["source_file"]
        file_extension_validation(source_file, {"py", "ipynb"})

        requirements_file = values["requirements_file"]
        file_extension_validation(requirements_file, {"txt"})

        keys = (
            "training_hash",
            "group",
            "model_type",
            "login",
            "password",
            "url",
        )

        data = {key: values[key] for key in keys}

        return data

    def _promote(self, source_file_path: str, schema_path: str, operation: str, model_name: str, input_type: str, model_reference: str):
        """
        Abstract method to promote the execution.

        Parameters
        ----------
        source_file_path: str
            A python script with an entry point function. It needs to return a dict, a list of dicts or a JSON string
        schema_path: str
            A JSON, CSV or PARQUET file with a sample of the input for the entry point function
        operation: str
            Defines how the model will be treated at the API. It can be: Sync or Async
        input_type: str
            The type of the input that the model expects, for example, .csv
        model_name: str
            Corresponds to the name of the model
        model_reference: str
            The name of the entry point function at the source file
        Returns
        -------
        str
            Model hash
        """
        user_token = refresh_token(
            *self.mlops_class.credentials, self.mlops_class.base_url
        )
        response = make_request(
            url=f"{self.url}/v2/training/execution/{self.execution_id}/promote",
            method="PATCH",
            success_code=201,
            headers={
                "Authorization": f"Bearer {user_token}"
            },
            files={
                "source": open(source_file_path, "rb"),
                "schema": open(schema_path, "rb"),
            },
            data={
                "operation": operation,
                "input_type": input_type,
                "name": model_name,
                "model_reference": model_reference,
            }
        )

        msg = response.json()["Message"]
        logger.info(msg)

        model_hash = response.json()["ModelHash"]
        logger.info(f"Model hash: {model_hash}")
        return model_hash

    def __init__(self, **data):
        super().__init__(**data)

        if data.get("is_copy", False):
            return

        user_token = refresh_token(
            *self.mlops_class.credentials, self.mlops_class.base_url
        )

        self.execution_id = trigger_custom_training(
            url=self.mlops_class.base_url,
            token=user_token,
            training_hash=self.training_hash,
            run_name=data["run_name"],
            description=data["description"],
            input_data=data["input_data"],
            upload_data=data["upload_data"],
            requirements_file=data["requirements_file"],
            source_file=data["source_file"],
            training_reference=data["training_reference"],
            python_version=data["python_version"],
            extra_files=data["extra_files"],
            env=data["env"],
        )

        self.host()

        if data["wait_complete"]:
            self.wait_ready()

    @classmethod
    def _do_copy(cls, url, token, group, experiment_name, mlops_class, **kwargs):
        """
        Abstract method to copy the execution.

        Parameters
        ----------
        url: str
            URL to copy the execution.
        token: str
            Authentication token.
        group: str
            Group where the training is inserted.
        experiment_name: str
            Name of the experiment.
        mlops_class: BaseMLOps
            MLOps class instance.
        kwargs: dict
            Extra arguments passed to the specific function.
        """
        response = make_request(
            url=url,
            method="POST",
            success_code=201,
            headers={"Authorization": f"Bearer {token}"},
        ).json()

        logger.info(response["Message"])

        fields = dict(
            training_hash=response["TrainingHash"],
            group=group,
            model_type="Custom",
            execution_id=response["ExecutionId"],
            experiment_name=experiment_name,
            login=mlops_class.credentials[0],
            password=mlops_class.credentials[1],
            url=mlops_class.base_url,
            mlops_class=mlops_class,
            is_copy=True,
        )

        new_execution = cls.model_construct(**fields)

        new_execution._update_execution(
            token=token,
            source_file=kwargs.get("source_file"),
            training_reference=kwargs.get("training_reference"),
            python_version=kwargs.get("python_version"),
            train_data=kwargs.get("train_data"),
            dataset_name=kwargs.get("dataset_name", "input"),
            dataset=kwargs.get("dataset"),
            requirements_file=kwargs.get("requirements_file"),
            extra_files=kwargs.get("extra_files", []),
            env=kwargs.get("env"),
            wait_complete=kwargs.get("wait_complete"),
        )

        return new_execution

    def _update_execution(
        self,
        token: str,
        source_file: str = None,
        training_reference: str = None,
        python_version: str = None,
        train_data: str = None,
        dataset_name: str = "input",
        dataset: Union[str, MLOpsDataset] = None,
        requirements_file: str = None,
        extra_files: str = None,
        env: str = None,
        wait_complete: bool = True,
    ):
        """
        Updates the execution with new parameters.

        Parameters
        ----------
        source_file : str, optional
            Path to the source code file, by default None
        training_reference : str, optional
            Training reference identifier, by default None
        python_version : str, optional
            Python version to use, by default None
        requirements_file : str, optional
            Path to requirements.txt file, by default None
        extra_files : str, optional
            List of additional files to include, by default None
        env : str, optional
            Environment variables, by default None
        wait_complete : bool, optional
            Whether to wait for execution completion, by default None

        Raises
        ------
        TrainingError
            If the execution is not in requested state.
        """

        if dataset is not None:
            dataset_hash = validate_dataset(dataset)
        else:
            dataset_hash = None

        input_data, upload_data = parse_data(
            file_path=train_data,
            form_data="dataset_hash" if dataset_hash is not None else "dataset_name",
            file_name=dataset_name,
            file_form="input",
            dataset_hash=dataset_hash,
        )

        self.execution_id = trigger_custom_training(
            url=self.mlops_class.base_url,
            token=token,
            execution_id=self.execution_id,
            input_data=input_data,
            upload_data=upload_data,
            requirements_file=requirements_file,
            source_file=source_file,
            training_reference=training_reference,
            python_version=python_version,
            extra_files=extra_files,
            env=env,
        )

        self.host()

        if wait_complete:
            self.wait_ready()


class AutoMLTrainingExecution(ITrainingExecution):
    """
    AutoML training execution class.

    Parameters
    ----------
    training_hash: str
        Training hash.
    group: str
        Group where the training is inserted.
    model_type: str
        Type of the model.
    execution_id: int
        Execution ID of a training.
    experiment_name: str
        Name of the experiment.
    login: str
        Login credential.
    password: str
        Password credential.
    url: str
        URL used to connect to the MLOps server.
    mlops_class: BaseMLOps
        MLOps class instance.
    """

    @model_validator(mode="before")
    @classmethod
    def validate(cls, values):
        """
        Validates the input values for AutoML training execution.

        Parameters
        ----------
        values: dict
            Dictionary of input values.

        Returns
        -------
        dict
            Validated input values.
        """

        validate_input({"input_data", "upload_data", "conf_dict", "run_name"}, values)

        file_extension_validation(values["conf_dict"], {"json"})

        keys = (
            "training_hash",
            "group",
            "model_type",
            "login",
            "password",
            "url",
        )

        data = {key: values[key] for key in keys}

        return data

    def _promote(self, model_name: str, input_type: str, operation: str, schema_path: str):
        """
        Abstract method to promote the execution.

        Parameters
        ----------
        schema_path: str
            A JSON, CSV or PARQUET file with a sample of the input for the entry point function
        operation: str
            Defines how the model will be treated at the API. It can be: Sync or Async
        model_name: str
            Corresponds to the name of the model
        input_type: str
            Type of the input that the model expects. Must be CSV or Parquet
        Returns
        -------
        str
            Model hash
        """

        raise NotImplementedError()

    def __init__(self, **data):
        super().__init__(**data)

        if data.get("is_copy", False):
            return

        user_token = refresh_token(
            *self.mlops_class.credentials, self.mlops_class.base_url
        )

        self.execution_id = trigger_automl_training(
            url=self.mlops_class.base_url,
            token=user_token,
            training_hash=self.training_hash,
            run_name=data["run_name"],
            description=data["description"],
            input_data=data["input_data"],
            upload_data=data["upload_data"],
            conf_dict=data["conf_dict"],
            extra_files=data["extra_files"],
        )

        self.host()

        if data["wait_complete"]:
            self.wait_ready()

    @classmethod
    def _do_copy(cls, url, token, group, experiment_name, mlops_class, **kwargs):
        """
        Abstract method to copy the execution.

        Parameters
        ----------
        url: str
            URL to copy the execution.
        token: str
            Authentication token.
        group: str
            Group where the training is inserted.
        experiment_name: str
            Name of the experiment.
        mlops_class: BaseMLOps
            MLOps class instance.
        kwargs: dict
            Extra arguments passed to the specific function.
        """
        response = make_request(
            url=url,
            method="POST",
            success_code=200,
            headers={"Authorization": f"Bearer {token}"},
        ).json()

        fields = dict(
            training_hash=response["TrainingHash"],
            group=group,
            model_type="AutoML",
            execution_id=response["ExecutionId"],
            experiment_name=experiment_name,
            login=mlops_class.credentials[0],
            password=mlops_class.credentials[1],
            url=mlops_class.base_url,
            mlops_class=mlops_class,
            is_copy=True,
        )

        new_execution = cls.model_construct(**fields)

        new_execution._update_execution(
            conf_dict=kwargs.get("conf_dict"),
            extra_files=kwargs.get("extra_files", []),
            train_data=kwargs.get("train_data"),
            dataset_name=kwargs.get("dataset_name", "input"),
            dataset=kwargs.get("dataset"),
            wait_complete=kwargs.get("wait_complete"),
        )

    def _update_execution(
        self,
        conf_dict: str = None,
        train_data: str = None,
        dataset_name: str = "input",
        dataset: Union[str, MLOpsDataset] = None,
        extra_files: str = None,
        wait_complete: bool = True,
    ):
        if dataset is not None:
            dataset_hash = validate_dataset(dataset)
        else:
            dataset_hash = None

        input_data, upload_data = parse_data(
            file_path=train_data,
            form_data="dataset_hash" if dataset_hash is not None else "dataset_name",
            file_name=dataset_name,
            file_form="input",
            dataset_hash=dataset_hash,
        )

        self.execution_id = trigger_automl_training(
            execution_id=self.execution_id,
            url=self.mlops_class.base_url,
            token=refresh_token(
                *self.mlops_class.credentials, self.mlops_class.base_url
            ),
            input_data=input_data,
            upload_data=upload_data,
            conf_dict=parse_dict_or_file(conf_dict),
            extra_files=extra_files,
        )

        self.host()

        if wait_complete:
            self.wait_ready()


class ExternalTrainingExecution(ITrainingExecution):
    """
    External training execution class.

    Parameters
    ----------
    training_hash: str
        Training hash.
    group: str
        Group where the training is inserted.
    model_type: str
        Type of the model.
    execution_id: int
        Execution ID of a training.
    experiment_name: str
        Name of the experiment.
    login: str
        Login credential.
    password: str
        Password credential.
    url: str
        URL used to connect to the MLOps server.
    mlops_class: BaseMLOps
        MLOps class instance.
    """

    @model_validator(mode="before")
    @classmethod
    def validate(cls, values):
        """
        Validates the input values for External training execution.

        Parameters
        ----------
        values: dict
            Dictionary of input values.

        Returns
        -------
        dict
            Validated input values.
        """
        logger.info("Validating external training execution...")

        copy_dict = {
            "run_name": values["run_name"],
            "features": (
                values.get("features_file")
                if values.get("features_file")
                else values.get("features_hash")
            ),
            "target": (
                values.get("target_file")
                if values.get("target_file")
                else values.get("target_hash")
            ),
            "output": (
                values.get("output_file")
                if values.get("output_file")
                else values.get("output_hash")
            ),
        }

        validate_input({"run_name", "features", "target", "output"}, copy_dict)

        if values["features_file"] and values["features_hash"]:
            raise InputError("You must provide either features file or dataset hash.")

        if values["output_file"] and values["output_hash"]:
            raise InputError("You must provide either output file or dataset hash.")

        if values["target_file"] and values["target_hash"]:
            raise InputError("You must provide either target file or dataset hash.")

        if values["requirements_file"]:
            file_extension_validation(values["requirements_file"], {"txt"})

        keys = (
            "training_hash",
            "group",
            "model_type",
            "login",
            "password",
            "url",
        )

        data = {key: values[key] for key in keys}

        return data

    def __init__(self, **data):
        super().__init__(**data)

        if data.get("is_copy", False):
            return

        user_token = refresh_token(
            *self.mlops_class.credentials, self.mlops_class.base_url
        )

        self.execution_id = trigger_external_training(
            url=self.mlops_class.base_url,
            token=user_token,
            training_hash=self.training_hash,
            run_name=data["run_name"],
            description=data["description"],
            features_file=data["features_file"],
            features_hash=data["features_hash"],
            target_file=data["target_file"],
            target_hash=data["target_hash"],
            output_file=data["output_file"],
            output_hash=data["output_hash"],
            metrics_file=data["metrics_file"],
            parameters_file=data["parameters_file"],
            model_file=data["model_file"],
            requirements_file=data["requirements_file"],
            python_version=data["python_version"],
        )

        self.host()

        if data["wait_complete"]:
            self.wait_ready()

    def _promote(self, *args, **kwargs):
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

    @classmethod
    def _do_copy(cls, url, token, group, experiment_name, mlops_class, **kwargs):
        """
        Abstract method to copy the execution.

        Parameters
        ----------
        url: str
            URL to copy the execution.
        token: str
            Authentication token.
        group: str
            Group where the training is inserted.
        experiment_name: str
            Name of the experiment.
        mlops_class: BaseMLOps
            MLOps class instance.
        kwargs: dict
            Extra arguments passed to the specific function.
        """
        response = make_request(
            url=url,
            method="POST",
            success_code=200,
            headers={"Authorization": f"Bearer {token}"},
        ).json()

        fields = dict(
            training_hash=response["TrainingHash"],
            group=group,
            model_type="External",
            execution_id=response["ExecutionId"],
            experiment_name=experiment_name,
            login=mlops_class.credentials[0],
            password=mlops_class.credentials[1],
            url=mlops_class.base_url,
            mlops_class=mlops_class,
            is_copy=True,
        )

        new_training = cls.model_construct(**fields)
        new_training._update_execution(
            features_file=kwargs.get("features_file"),
            features_hash=kwargs.get("features_hash"),
            target_file=kwargs.get("target_file"),
            target_hash=kwargs.get("target_hash"),
            output_file=kwargs.get("output_file"),
            output_hash=kwargs.get("output_hash"),
            metrics_file=kwargs.get("metrics_file"),
            parameters_file=kwargs.get("parameters_file"),
            model_file=kwargs.get("model_file"),
            requirements_file=kwargs.get("requirements_file"),
            python_version=kwargs.get("python_version", "3.10"),
            wait_complete=kwargs.get("wait_complete"),
        )

        return new_training

    def _update_execution(
        self,
        features_file: str = None,
        features_hash: str = None,
        target_file: str = None,
        target_hash: str = None,
        output_file: str = None,
        output_hash: str = None,
        metrics_file: str = None,
        parameters_file: str = None,
        model_file: str = None,
        requirements_file: str = None,
        python_version: str = None,
        wait_complete: bool = True,
    ):
        """
        Updates the execution with new parameters.

        Parameters
        ----------
        features_file : str, optional
            Path to features file, by default None
        features_hash : str, optional
            Features dataset hash, by default None
        target_file : str, optional
            Path to target file, by default None
        target_hash : str, optional
            Target dataset hash, by default None
        output_file : str, optional
            Path to output file, by default None
        output_hash : str, optional
            Output dataset hash, by default None
        metrics_file : str, optional
            Path to metrics file, by default None
        parameters_file : str, optional
            Path to parameters file, by default None
        model_file : str, optional
            Path to model file, by default None
        requirements_file : str, optional
            Path to requirements file, by default None
        python_version : str, optional
            Python version to use, by default None
        wait_complete : bool, optional
            Whether to wait for execution completion, by default True
        """

        self.execution_id = trigger_external_training(
            execution_id=self.execution_id,
            url=self.mlops_class.base_url,
            token=refresh_token(
                *self.mlops_class.credentials, self.mlops_class.base_url
            ),
            features_file=features_file,
            features_hash=features_hash,
            target_file=target_file,
            target_hash=target_hash,
            output_file=output_file,
            output_hash=output_hash,
            metrics_file=metrics_file,
            parameters_file=parameters_file,
            model_file=model_file,
            requirements_file=requirements_file,
            python_version=python_version,
        )

        self.host()

        if wait_complete:
            self.wait_ready()
