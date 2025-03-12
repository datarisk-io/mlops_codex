from pydantic import model_validator

from mlops_codex.__model_states import ModelTypes
from mlops_codex.__utils import parse_dict_or_file
from mlops_codex.exceptions import InputError
from mlops_codex.http_request_handler import make_request, refresh_token
from mlops_codex.logger_config import get_logger
from mlops_codex.training.base import ITrainingExecution
from mlops_codex.validations import file_extension_validation, validate_python_version

logger = get_logger()


class CustomTrainingExecution(ITrainingExecution):

    @model_validator(mode="before")
    @classmethod
    def validate(cls, values):

        fields_required = (
            "input_data",
            "upload_data",
            "run_name",
            "source_file",
            "requirements_file",
            "training_reference",
            "python_version",
        )

        if (not all(k in values for k in fields_required)) or (
            not all(values[f] for f in fields_required)
        ):
            raise InputError(
                f"The parameters {fields_required} it's mandatory on custom training."
            )

        python_version = validate_python_version(values["python_version"])

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
        data["python_version"] = python_version

        return data

    # This function is not accessible to users. When the promote endpoint is changed, I'll make some updates here
    def __promote(
        self,
        model_name: str,
        operation: str,
        source_file: str,
        model_reference: str,
        schema: str,
        requirements_file: str = None,
        input_type: str = None,
        env: str = None,
        extra_files: list = None,
        wait_complete: bool = False,
    ):
        schema_extension = schema.split(".")[-1]
        self._promote_validation(
            operation=operation,
            input_type=input_type,
            schema_extension=schema_extension,
        )
        operation = ModelTypes[operation.title()]

        file_extension_validation(source_file, {"py", "ipynb"})

        file_extension_validation(schema, {"json", "xml", "csv", "parquet"})
        file_extension_validation(requirements_file, {"txt"})
        file_extension_validation(env, {"env"})

        input_data = {
            "name": model_name,
            "operation": operation,
            "schema": schema,
            "model_reference": model_reference,
            "input_type": input_type,
        }

        upload_data = [
            ("source", ("app.py", open(source_file, "rb"))),
            ("schema", (f"schema.{schema_extension}", parse_dict_or_file(schema))),
        ]
        if env is not None:
            upload_data.append(("env", (".env", open(env, "rb"))))

        if requirements_file is not None:
            upload_data.append(
                ("requirements", ("requirements.txt", open(requirements_file, "rb")))
            )

        if extra_files is not None:
            extra_data = [
                ("extra", (c.split("/")[-1], open(c, "rb"))) for c in extra_files
            ]

            upload_data += extra_data

        self._promote(
            upload_data=upload_data,
            input_data=input_data,
            operation=operation,
            model_name=model_name,
            wait_complete=wait_complete,
        )

    def promote(self, *args, **kwargs):
        raise NotImplementedError()

    def __upload_script_file(
        self, script_path: str, train_reference: str, python_version: str
    ):
        url = f"{self.mlops_class.base_url}/v2/training/execution/{self.execution_id}/script-file"
        token = refresh_token(*self.mlops_class.credentials, self.mlops_class.base_url)
        upload_data = {"script": open(script_path, "rb")}
        input_data = {
            "train_reference": train_reference,
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
        self.execution_id = self._register_execution(
            run_name=data["run_name"],
            description=data["description"],
            training_type="Custom",
        )

        self._upload_input_file(
            input_data=data["input_data"], upload_data=data["upload_data"]
        )

        self._upload_requirements(requirements_file=data["requirements_file"])


        self.__upload_script_file(
            script_path=data["source_file"],
            train_reference=data["train_reference"],
            python_version=data["python_version"],
        )

        for path, name in data["extras"]:
            self._upload_extra_files(extra_files_path=path, name=name)

        if data["env"]:
            self._upload_env_file(env_file=data["env"])

        self.host()

        if data["wait_complete"]:
            self.wait_ready()


class AutoMLTrainingExecution(ITrainingExecution):
    @model_validator(mode="before")
    def validate(self, values):
        fields_required = (
            "train_data",
            "conf_dict",
            "run_name",
            "description",
        )

        if (not all(k in values for k in fields_required)) or (
            not all(v for v in values.values())
        ):
            raise InputError(
                f"The parameters {fields_required} it's mandatory on automl training."
            )

        file_extension_validation(values["conf_dict"], {"json"})

        self.execution_id = self._register_execution(
            run_name=values["run_name"],
            description=values["description"],
            training_type="AutoML",
        )

        self.__upload_conf_dict(conf_dict=values["conf_dict"])

        self._upload_input_file(
            input_data=values["input_data"], upload_data=values["upload_data"]
        )

        for path, name in values["extras"]:
            self._upload_extra_files(extra_files_path=path, name=name)

        self.host()

        if values["wait_complete"]:
            self.wait_ready()

    def __upload_conf_dict(self, conf_dict):
        url = f"{self.mlops_class.base_url}/v2/training/execution/{self.execution_id}/confi-dict/file"
        token = refresh_token(*self.mlops_class.credentials, self.mlops_class.base_url)

        upload_data = {"conf_dict": parse_dict_or_file(conf_dict)}
        response = make_request(
            url=url,
            method="PATCH",
            success_code=201,
            files=upload_data,
            headers={
                "Authorization": f"Bearer {token}",
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.__upload_conf_dict.__qualname__,
            },
        ).json()

        msg = response["Message"]
        logger.info(msg)

    def promote(self, *args, **kwargs):
        raise NotImplementedError()
