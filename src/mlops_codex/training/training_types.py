from typing import Union

from pydantic import model_validator

from mlops_codex.__utils import parse_dict_or_file
from mlops_codex.dataset import MLOpsDataset, validate_dataset
from mlops_codex.exceptions import InputError
from mlops_codex.http_request_handler import make_request, refresh_token
from mlops_codex.logger_config import get_logger
from mlops_codex.shared.utils import parse_data
from mlops_codex.training.base import ITrainingExecution
from mlops_codex.training.trigger import (
    trigger_automl_training,
    trigger_custom_training,
    trigger_external_training,
)
from mlops_codex.training.validations import validate_input
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

    def __upload_script_file(
        self, script_path: str, train_reference: str, python_version: str
    ):
        """
        Uploads the script file.

        Parameters
        ----------
        script_path: str
            Path to the script file.
        train_reference: str
            Training reference.
        python_version: str
            Python version.

        Returns
        -------
        None
        """

        url = f"{self.mlops_class.base_url}/v2/training/execution/{self.execution_id}/script-file"
        token = refresh_token(*self.mlops_class.credentials, self.mlops_class.base_url)
        upload_data = {"script": open(script_path, "rb")}
        input_data = {
            "training_reference": train_reference,
            "python_version": python_version,
        }
        response = make_request(
            url=url,
            method="PATCH",
            success_code=201,
            data=input_data,
            files=upload_data,
            headers={
                "Authorization": f"Bearer {token}",
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.__upload_script_file.__qualname__,
            },
        )

        msg = response.json()["Message"]
        logger.info(msg)

    def __init__(self, **data):
        super().__init__(**data)

        if data.get('is_copy', False):
            return

        self.execution_id = self._register_execution(
            run_name=data["run_name"],
            description=data["description"],
            training_type="Custom",
        )

        self._upload_input_file(
            input_data=data["input_data"], upload_data=data["upload_data"]
        )

        self._upload_requirements(requirements_file=data["requirements_file"])

        python_version = validate_python_version(data["python_version"])

        self.__upload_script_file(
            script_path=data["source_file"],
            train_reference=data["training_reference"],
            python_version=python_version,
        )

        for name, path in data["extra_files"]:
            self._upload_extra_files(extra_files_path=path, name=name)

        if data["env"]:
            self._upload_env_file(env_file=data["env"])

        self.host()

        if data["wait_complete"]:
            self.wait_ready()

    @classmethod
    def _do_copy(cls, url, token, group, experiment_name, mlops_class):
        response = make_request(
            url=url,
            method="POST",
            success_code=201,
            headers={
                "Authorization": f"Bearer {token}"
            },
        ).json()

        fields = dict(
            training_hash=response["TrainingHash"],
            group=group,
            model_type='Custom',
            execution_id=response["ExecutionId"],
            experiment_name=experiment_name,
            login=mlops_class.credentials[0],
            password=mlops_class.credentials[1],
            url=mlops_class.base_url,
            mlops_class=mlops_class,
            is_copy=True,
        )

        return cls.model_construct(**fields)



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

    def __init__(self, **data):
        super().__init__(**data)

        if data.get('is_copy', False):
            return

        self.execution_id = self._register_execution(
            run_name=data["run_name"],
            description=data["description"],
            training_type="AutoML",
        )

        self.__upload_conf_dict(conf_dict=data["conf_dict"])

        self._upload_input_file(
            input_data=data["input_data"], upload_data=data["upload_data"]
        )

        for name, path in data["extra_files"]:
            self._upload_extra_files(extra_files_path=path, name=name)

        self.host()

        if data["wait_complete"]:
            self.wait_ready()

    @classmethod
    def _do_copy(cls, url, token, group, experiment_name, mlops_class):
        response = make_request(
            url=url,
            method="POST",
            success_code=200,
            headers={
                "Authorization": f"Bearer {token}"
            },
        ).json()

        fields = dict(
            training_hash=response["TrainingHash"],
            group=group,
            model_type='AutoML',
            execution_id=response["ExecutionId"],
            experiment_name=experiment_name,
            login=mlops_class.credentials[0],
            password=mlops_class.credentials[1],
            url=mlops_class.base_url,
            mlops_class=mlops_class,
            is_copy=True,
        )

        return cls.model_construct(**fields)


class ExternalTrainingExecution(ITrainingExecution):
    @model_validator(mode="before")
    @classmethod
    def validate(cls, values):
        logger.info("Validating external training execution...")

        copy_dict = {
            "run_name": values["run_name"],
            "features": values.get("features_file")
            if values.get("features_file")
            else values.get("features_hash"),
            "target": values.get("target_file")
            if values.get("target_file")
            else values.get("target_hash"),
            "output": values.get("output_file")
            if values.get("output_file")
            else values.get("output_hash"),
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

        if data.get('is_copy', False):
            return

        self.execution_id = self._register_execution(
            run_name=data["run_name"],
            description=data["description"],
            training_type="External",
        )

        for var in ["features", "target", "output"]:
            inp = data[f"{var}_hash"] if data[f"{var}_hash"] is not None else var
            form = "dataset_hash" if data[f"{var}_hash"] is not None else "dataset_name"
            file = (
                open(data[f"{var}_file"], "rb")
                if data[f"{var}_file"] is not None
                else var
            )
            self.__upload_file_or_hash(
                url=var, input_data={form: inp}, upload_data={var: file}
            )

        if data["metrics_file"]:
            self.__upload_file_or_hash(
                url="metrics", upload_data={"metrics": open(data["metrics_file"], "rb")}
            )
        if data["parameters_file"]:
            self.__upload_file_or_hash(
                url="parameters",
                upload_data={"parameters": open(data["parameters_file"], "rb")},
            )
        if data["model_file"]:
            self.__upload_file_or_hash(
                url="model", upload_data={"model": open(data["model_file"], "rb")}
            )
        if data["requirements_file"]:
            self._upload_requirements(requirements_file=data["requirements_file"])

        python_version = validate_python_version(data["python_version"])
        self.__set_python_version(python_version)

        self.host()

        if data["wait_complete"]:
            self.wait_ready()

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

    @classmethod
    def _do_copy(cls, url, token, group, experiment_name, mlops_class):
        response = make_request(
            url=url,
            method="POST",
            success_code=200,
            headers={
                "Authorization": f"Bearer {token}"
            },
        ).json()

        fields = dict(
            training_hash=response["TrainingHash"],
            group=group,
            model_type='External',
            execution_id=response["ExecutionId"],
            experiment_name=experiment_name,
            login=mlops_class.credentials[0],
            password=mlops_class.credentials[1],
            url=mlops_class.base_url,
            mlops_class=mlops_class,
            is_copy=True,
        )

        return cls.model_construct(**fields)