#!/usr/bin/env python
# coding: utf-8

import json
import os
from time import sleep
from typing import List, Optional, Tuple, Union

from mlops_codex.__model_states import ModelExecutionState, ModelState, MonitoringStatus
from mlops_codex.__utils import (
    parse_dict_or_file,
    parse_json_to_yaml,
)
from mlops_codex.base import BaseMLOps, BaseMLOpsClient, MLOpsExecution
from mlops_codex.datasources import MLOpsDataset
from mlops_codex.exceptions import (
    InputError,
    ModelError,
    PreprocessingError,
)
from mlops_codex.http_request_handler import make_request, refresh_token
from mlops_codex.logger_config import get_logger
from mlops_codex.preprocessing import MLOpsPreprocessing
from mlops_codex.validations import (
    file_extension_validation,
    validate_data,
    validate_group_existence,
    validate_python_version,
)

logger = get_logger()


def _model_status(url, credentials, group, model_hash):
    """Get the status of a model

    Args:
        url: Url used to connect to the MLOps server
        credentials: User credentials
        group: Group where the model is located
        model_hash: Hash of the model

    Returns:
        ModelState: Status of the model
    """
    token = refresh_token(*credentials, url)

    response = make_request(
        url=f"{url}/model/status/{group}/{model_hash}",
        method="GET",
        success_code=200,
        custom_exception=ModelError,
        custom_exception_message=f"Model with hash {model_hash} not found for group {group}.",
        specific_error_code=404,
        logger_msg=f"Model with hash {model_hash} not found for group {group}.",
        headers={
            "Authorization": f"Bearer {token}",
            "Neomaril-Origin": "Codex",
            "Neomaril-Method": _model_status.__qualname__,
        },
    ).json()

    status = response["Status"]

    if status == "Failed":
        logger.info(f"Model failed. Reason: {response['Message']}")

    return ModelState[status]


class MLOpsModel(BaseMLOps):
    """
    Class to manage Models deployed inside MLOps

    Parameters
    ----------
    login: str
        Login for authenticating with the client. You can also use the env variable MLOPS_USER to set this
    password: str
        Password for authenticating with the client. You can also use the env variable MLOPS_PASSWORD to set this
    model_hash: str
        Model id (hash) from the model you want to access
    group: str
        Group the model is inserted.
    group_token: str
        Token for executing the model (show when creating a group). It can be informed when getting the model or when running predictions, or using the env variable MLOPS_GROUP_TOKEN
    url: str
        URL to MLOps Server. Default value is https://neomaril.datarisk.net/, use it to test your deployment first before changing to production. You can also use the env variable MLOPS_URL to set these

    Raises
    ------
    ModelError
        When the model can't be accessed in the server
    AuthenticationError
        Invalid credentials
    """

    def __init__(
        self,
        *,
        name: str,
        model_hash: str,
        group: str,
        login: Optional[str] = None,
        password: Optional[str] = None,
        group_token: Optional[str] = None,
        url: Optional[str] = None,
    ) -> None:
        super().__init__(login=login, password=password, url=url)

        self.model_hash = model_hash
        self.group = group
        self.group_token = (
            group_token if group_token else os.getenv("MLOPS_GROUP_TOKEN")
        )
        self.name = name

    def __repr__(self) -> str:
        return f"MLOpsModel(name={self.name}, model_hash={self.model_hash}, group={self.group}, group_token={self.group_token})"

    def __str__(self):
        return f"MLOpsModel(name={self.name}, model_hash={self.model_hash}, group={self.group}, group_token={self.group_token})"

    def host(self, operation):
        """
        Builds the model execution environment

        Parameters
        ----------
        operation: str
            The model operation type (Sync or Async)

        Raises
        ------
        InputError
            Some input parameters is invalid
        """
        logger.info(f"MLOpsModel hosting {self.name}...")
        token = refresh_token(*self.credentials, self.base_url)
        _ = make_request(
            url=f"{self.base_url}/model/{operation.lower()}/host/{self.group}/{self.model_hash}",
            method="GET",
            success_code=202,
            headers={
                "Authorization": f"Bearer {token}",
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.host.__qualname__,
            },
        )

    def _describe(self):
        """
        Get a description of the model

        Returns:
            dict: Description of the model
        """
        url = f"{self.base_url}/model/describe/{self.group}/{self.model_hash}"
        token = refresh_token(*self.credentials, self.base_url)
        response = make_request(
            url=url,
            method="GET",
            success_code=200,
            custom_exception=ModelError,
            custom_exception_message=f"Model with hash {self.model_hash} not found for group {self.group}.",
            specific_error_code=404,
            logger_msg=f"Model with hash {self.model_hash} not found for group {self.group}.",
            headers={
                "Authorization": f"Bearer {token}",
            },
        ).json()
        return response

    def _describe_execution(self, execution_id: Union[str, int]):
        """
        Get a description of the model execution

        Args:
            execution_id: Execution id of the model

        Returns:
            dict: Description of the model execution
        """
        url = f"{self.base_url}/model/describe/{self.group}/{self.model_hash}/{execution_id}"
        token = refresh_token(*self.credentials, self.base_url)

        response = make_request(
            url=url,
            method="GET",
            success_code=200,
            custom_exception=ModelError,
            custom_exception_message=f"Model with hash {self.model_hash} and execution id {execution_id} not found for group {self.group}.",
            specific_error_code=404,
            logger_msg=f"Model with hash {self.model_hash} and execution id {execution_id} not found for group {self.group}.",
            headers={
                "Authorization": f"Bearer {token}",
            },
        ).json()
        return response

    def status(self):
        """
        Gets the status of the model.

        Raises
        -------
        ModelError
            Execution unavailable

        Returns
        -------
        str
            The model status
        """
        return _model_status(
            self.base_url, self.credentials, self.group, self.model_hash
        )

    def wait_ready(self):
        """
        Waits the model to be with status different from Ready or Building
        """
        current_status = ModelState.Building
        print("Waiting for model finish building...", end="", flush=True)
        while current_status in [ModelState.Ready, ModelState.Building]:
            sleep(30)
            current_status = self.status()
            print(".", end="", flush=True)
        print()

        if current_status == ModelState.Deployed:
            logger.info("Model deployed successfully")
        else:
            logger.info(f"Model deployed failed. Status {current_status}")

    def health(self):
        """
        Get the model health state.

        Raises
        ------
        AuthenticationError
            Raised if there is an authentication issue.
        ServerError
            Raised if the server encounters an issue.
        ModelError
            Raised if it can not get the health of the model
        """
        url = f"{self.base_url}/model/sync/health/{self.group}/{self.model_hash}"
        response = make_request(
            url=url,
            method="GET",
            success_code=200,
            custom_exception=ModelError,
            custom_exception_message=f"Model with hash {self.model_hash} not found.",
            specific_error_code=404,
            logger_msg=f"Model with hash {self.model_hash} not found.",
            headers={
                "Authorization": "Bearer " + self.group_token,
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.health.__qualname__,
            },
        ).json()["Message"]

        if response == "OK":
            logger.info("Model is healthy")
        else:
            logger.info(
                "Model is not healthy. If you wish to use this model, consider using the 'restart_model' function."
            )

    def restart_model(self, wait_for_ready: bool = True):
        """
        Restart a model deployment process health state. The model will be restarted if the state is one of following states:
            - Deployed;
            - Disabled;
            - DisabledRecovery;
            - FailedRecovery.

        Parameters
        -----------
        wait_for_ready: bool, default=True
            If the model is being deployed, wait for it to be ready instead of failing the request

        Raises
        ------
        AuthenticationError
            Raised if there is an authentication issue.
        ServerError
            Raised if the server encounters an issue.
        ModelError
            Raised if model could not be restarted.
        """

        current_status = self.status()
        if current_status not in [
            ModelState.Deployed,
            ModelState.Disabled,
            ModelState.DisabledRecovery,
            ModelState.FailedRecovery,
        ]:
            logger.info(
                f"Model can't be restarted because it is current {current_status}"
            )
            return None

        url = f"{self.base_url}/model/restart/{self.group}/{self.model_hash}"
        token = refresh_token(*self.credentials, self.base_url)

        _ = make_request(
            url=url,
            method="GET",
            success_code=200,
            custom_exception=ModelError,
            custom_exception_message=f"Model with hash {self.model_hash} not found.",
            specific_error_code=404,
            logger_msg=f"Model with hash {self.model_hash} not found.",
            headers={
                "Authorization": f"Bearer {token}",
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.restart_model.__qualname__,
            },
        )

        logger.info("Model is restarting...")
        if wait_for_ready:
            self.wait_ready()

    def get_logs(
        self,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        routine: Optional[str] = None,
        log_type: Optional[str] = None,
    ):
        """
        Get the logs

        Parameters
        -----------
        start: Optional[str], default=None
            Date to start filter. At the format aaaa-mm-dd
        end: Optional[str], default=None
            Date to end filter. At the format aaaa-mm-dd
        routine: Optional[str], default=None
            Type of routine being executed, can assume values Host or Run
        log_type: Optional[str], default=None
            Defines the type of the logs that are going to be filtered, can assume the values Ok, Error, Debug or Warning

        Raises
        ------
        ServerError
            Unexpected server error

        Returns
        -------
        dict
            Logs list
        """
        url = f"{self.base_url}/model/logs/{self.group}/{self.model_hash}"
        logs_result = self._logs(
            url=url,
            credentials=self.credentials,
            start=start,
            end=end,
            routine=routine,
            type=log_type,
        )
        formatted_logs = parse_json_to_yaml(logs_result)
        print(formatted_logs)

    def delete(self):
        """
        Deletes the current model.

        Raises
        ------
        AuthenticationError
            Raised if there is an authentication issue.
        ServerError
            Model deleting failed
        """

        user_input = input(
            "Are you sure you want to delete this model? [Type the name of the model to delete]"
        )

        if user_input != self.name:
            logger.info(
                f"Model deletion failed. {user_input} is not the name of this model."
            )
            return None

        logger.warning(
            "This is irreversible, if you want to use the model again later you will need to upload again (and it will have a new hash)."
        )

        token = refresh_token(*self.credentials, self.base_url)

        _ = make_request(
            url=f"{self.base_url}/model/delete/{self.group}/{self.model_hash}",
            method="DELETE",
            success_code=200,
            custom_exception=ModelError,
            custom_exception_message=f"Model with hash {self.model_hash} not found.",
            specific_error_code=404,
            logger_msg=f"Model with hash {self.model_hash} not found.",
            headers={
                "Authorization": "Bearer " + token,
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.delete.__qualname__,
            },
        )

        logger.info(f"Model with hash {self.model_hash} deleted.")

        self._describe()

    def disable(self):
        """
        Disables a model. It means that you won't be able to perform some operations in the model
        Please, check with your team if you're allowed to perform this operation

        Raises
        ------
        AuthenticationError
            Raised if there is an authentication issue.
        ServerError
            Model disable failed
        """

        token = refresh_token(*self.credentials, self.base_url)

        _ = make_request(
            url=f"{self.base_url}/model/disable/{self.group}/{self.model_hash}",
            method="POST",
            success_code=200,
            custom_exception=ModelError,
            custom_exception_message=f"Model with hash {self.model_hash} not found.",
            specific_error_code=404,
            logger_msg=f"Model with hash {self.model_hash} not found.",
            headers={
                "Authorization": "Bearer " + token,
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.disable.__qualname__,
            },
        )

        logger.info(
            f"Model with hash {self.model_hash} disabled. If you wish to use this model, consider using the 'restart_model' function."
        )

        self._describe()

    def set_token(self, group_token: str) -> None:
        """
        Saves the group token for this model instance.

        Parameters
        ----------
        group_token: str
            Token for executing the model (show when creating a group). You can set this using the MLOPS_GROUP_TOKEN env variable
        """
        self.group_token = group_token

    def info(self) -> None:
        """Show the model data in a better format"""
        describe = self._describe()
        formatted_response = parse_json_to_yaml(describe)
        logger.info(f"Result:\n{formatted_response}")

    def execution_info(self, execution_id: str):
        """Show the model execution data in a better format"""
        describe = self._describe_execution(execution_id)
        formatted_response = parse_json_to_yaml(describe)
        logger.info(f"Result:\n{formatted_response}")

    def host_monitoring_status(self, period: str):
        """
        Get the host status for the monitoring configuration

        Parameters
        ----------
        period: str
            The monitoring period (Day, Week, Month)

        Raises
        ------
        ExecutionError
            Monitoring host failed
        ServerError
            Unexpected server error
        """

        url = (
            f"{self.base_url}/monitoring/status/{self.group}/{self.model_hash}/{period}"
        )
        token = refresh_token(*self.credentials, self.base_url)

        response = make_request(
            url=url,
            method="GET",
            success_code=200,
            headers={
                "Authorization": f"Bearer {token}",
            },
        ).json()

        status = MonitoringStatus[response["Status"]]
        if status == MonitoringStatus.Invalidated:
            msg = response["Message"]
            logger.info(f"Monitoring host {self.model_hash} invalidated. Reason: {msg}")

        return status

    def host_monitoring(self, period: str):
        """
        Host the monitoring configuration

        Parameters
        ----------
        period: str
            The monitoring period (Day, Week, Month)

        Raises
        ------
        InputError
            Monitoring host error
        """

        logger.info(f"Monitoring host for {self.model_hash} model started.")

        url = f"{self.base_url}/monitoring/host/{self.group}/{self.model_hash}/{period}"
        token = refresh_token(*self.credentials, self.base_url)

        _ = make_request(
            url=url,
            method="GET",
            success_code=200,
            headers={
                "Authorization": f"Bearer {token}",
            },
        )

    def wait_monitoring(self, period: str):
        """
        Wait for the monitoring configuration

        Parameters
        ----------
        period (str): Period of monitoring. It must be 'Day', 'Week' or 'Month'
        """
        current_status = MonitoringStatus.Validating
        print("Waiting for monitoring host to finish...", end="", flush=True)
        while current_status in [
            MonitoringStatus.Unvalidated,
            MonitoringStatus.Validating,
        ]:
            sleep(30)
            current_status = self.host_monitoring_status(period)
            print(".", end="", flush=True)
        print()

        if current_status == MonitoringStatus.Validated:
            logger.info("Model monitoring host finished successfully.")
        else:
            logger.info(
                f"Model monitoring host finished unsuccessfully. Status {current_status}"
            )
            raise ModelError("Model monitoring host failed.")

    def register_monitoring(
        self,
        *,
        preprocess_reference: str,
        shap_reference: str,
        configuration_file: Union[str, dict],
        preprocess_file: Optional[str] = None,
        requirements_file: Optional[str] = None,
        wait_complete: Optional[bool] = False,
    ) -> str:
        """
        Register the model monitoring configuration at the database

        Parameters
        ----------
        preprocess_reference: str
            Name of the preprocess reference
        shap_reference: str
            Name of the preprocess function
        configuration_file: str or dict
            Path of the configuration file in json format. It can also be a dict
        preprocess_file: Optional[str], default=None
            Path of the preprocess script
        requirements_file: Optional[str], default=None
            Path of the requirements file
        wait_complete: bool, default=False
            If it is True, wait until the monitoring host is Deployed or Failed

        Raises
        ------
        InputError
            Invalid parameters for model creation

        Returns
        -------
        str
            Model id (hash)
        """

        logger.info("Registering model monitoring configuration")
        conf = parse_dict_or_file(configuration_file)

        if isinstance(configuration_file, str):
            with open(configuration_file, "rb") as f:
                configuration_file = json.load(f)

        period = configuration_file["Period"]

        if period.title() not in ("Day", "Week", "Month", "Year"):
            raise InputError(
                "Invalid period. Expected 'Day', 'Week', 'Month' or 'Year'"
            )

        period = period.title()

        upload_data = [
            ("configuration", ("configuration.json", conf)),
        ]

        form_data = {
            "preprocess_reference": preprocess_reference,
            "shap_reference": shap_reference,
        }

        if preprocess_file:
            upload_data.append(
                (
                    "source",
                    (
                        "preprocess." + preprocess_file.split(".")[-1],
                        open(preprocess_file, "rb"),
                    ),
                )
            )

        if requirements_file:
            upload_data.append(
                ("requirements", ("requirements.txt", open(requirements_file, "rb")))
            )

        url = f"{self.base_url}/monitoring/register/{self.group}/{self.model_hash}"
        token = refresh_token(*self.credentials, self.base_url)
        response = make_request(
            url=url,
            method="POST",
            data=form_data,
            files=upload_data,
            success_code=201,
            headers={
                "Authorization": f"Bearer {token}",
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.register_monitoring.__qualname__,
            },
        ).json()

        model_id = response["ModelHash"]
        logger.info(f'{response["Message"]} - Hash: "{model_id}"')

        self.host_monitoring(period=period)

        if wait_complete:
            self.wait_monitoring(period=period)

        logger.info("Model monitoring host finished successfully.")


class SyncModel(MLOpsModel):
    def __init__(
        self,
        name: str,
        model_hash: str,
        group: str,
        login: Optional[str] = None,
        password: Optional[str] = None,
        group_token: Optional[str] = None,
        url: Optional[str] = None,
    ):
        super().__init__(
            name=name,
            model_hash=model_hash,
            group=group,
            login=login,
            password=password,
            group_token=group_token,
            url=url,
        )

    def predict(
        self,
        json_data: Union[str, dict],
        preprocessing: Optional[MLOpsPreprocessing] = None,
        group_token: Optional[str] = None,
    ):
        """
        Run the hosted model for a specific input. It will show the result of the prediction

        Parameters
        ----------
        json_data: Union[str, dict]
            Input file that will be used to run the model. It must be a dict or a json file
        preprocessing: MLOpsPreprocessing, default=None
            Class for preprocessing json_data
        group_token: str, default=None
            Token of the group
        """
        if group_token:
            self.set_token(group_token)

        logger.info("Validating data...")

        if isinstance(json_data, str) and json_data.endswith(".json"):
            with open(json_data, "r", encoding="utf-8") as f:
                json_data = json.load(f)

        upload_data = {"Input": json_data}

        if preprocessing:
            logger.info("Found preprocessing...")
            upload_data["ScriptHash"] = preprocessing.preprocessing_id

        logger.info("Running data prediction...")

        response = make_request(
            url=f"{self.base_url}/model/sync/run/{self.group}/{self.model_hash}",
            method="POST",
            data=json.dumps(upload_data),
            success_code=200,
            custom_exception=ModelError,
            custom_exception_message=f"Failed to predict data for model {self.model_hash} in group {self.group}",
            specific_error_code=404,
            logger_msg=f"Failed to predict data for model {self.model_hash} in group {self.group}",
            headers={
                "Authorization": f"Bearer {self.group_token}",
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.predict.__qualname__,
            },
        ).json()

        logger.info("Prediction result:\n")
        formated_response = parse_json_to_yaml(response)
        print(formated_response)

    def __call__(
        self,
        json_data: Union[str, dict],
        preprocessing: MLOpsPreprocessing = None,
        group_token=None,
    ):
        self.predict(json_data, preprocessing, group_token)


class AsyncModel(MLOpsModel):
    def __init__(
        self,
        name: str,
        model_hash: str,
        group: str,
        login: Optional[str] = None,
        password: Optional[str] = None,
        group_token: Optional[str] = None,
        url: Optional[str] = None,
    ):
        super().__init__(
            name=name,
            model_hash=model_hash,
            group=group,
            login=login,
            password=password,
            group_token=group_token,
            url=url,
        )

    def execution_status(
        self, execution_id: Union[int, str], group_token: Optional[str] = None
    ):
        """
        Get the execution status of the model

        Parameters
        ----------
        execution_id: Union[int, str]
            Execution id of a model prediction
        group_token: Optional[str], default=None
            Token of the group

        Returns
        -------
        ModelExecutionStatus
            Status of the execution
        """
        if group_token:
            self.set_token(group_token)

        url = f"{self.base_url}/model/async/status/{self.group}/{execution_id}"
        response = make_request(
            url=url,
            method="GET",
            success_code=200,
            headers={
                "Authorization": f"Bearer {self.group_token}",
            },
        ).json()

        status = ModelExecutionState[response["Status"]]
        if status == ModelExecutionState.Failed:
            msg = response["Message"]
            logger.info(f"{msg} - Status: {status}")
        return status

    def wait_run_ready(
        self, execution_id: Union[int, str], group_token: Optional[str] = None
    ):
        """
        Loop until the model is ready to run

        Parameters
        ----------
        execution_id: Union[int, str]
            Execution id of a model prediction
        group_token: Optional[str], default=None
            Token of the group
        """
        current_status = ModelExecutionState.Running
        print("Waiting for model finish prediction...", end="", flush=True)
        while current_status in [
            ModelExecutionState.Requested,
            ModelExecutionState.Running,
        ]:
            sleep(30)
            current_status = self.execution_status(execution_id, group_token)
            print(".", end="", flush=True)
        print()

        if current_status == ModelExecutionState.Succeeded:
            logger.info("Prediction succeeded")
        else:
            logger.info(f"Prediction failed. Status {current_status}")

    def predict(
        self,
        data: Union[str, Tuple[str, str], List[Tuple[str, str]], MLOpsDataset, List[MLOpsDataset]],
        preprocessing: MLOpsPreprocessing = None,
        group_token=None,
        wait_complete: bool = True,
    ):
        """
        Run the hosted model for a specific input. It will show the result of the prediction

        Parameters
        ----------
        data: str | tuple[str, str] | list[tuple[str, str]] | MLOpsDataset | None
            Data that will be used to run the model. You can upload a dataset hash as string, a tuple with file name and file path,
            a list of tuples with file name and file path, a MLOpsDataset or a list of MLOpsDataset.
            If you provide a single string, it will consider it as a dataset hash.
        preprocessing: MLOpsPreprocessing, default=None
            Class for preprocessing data.
        group_token: str, default=None
            Token of the group
        wait_complete: bool, default=True
            Wait for model to be ready and returns a MLOpsModel instance with the new model.

        Returns
        -------
        ModelExecution
            Class to handle model execution
        """

        if group_token:
            self.set_token(group_token)

        logger.info("Validating data...")

        validate_data(data, {"csv", "parquet"})

        if preprocessing:
            logger.info("Preprocessing data...")
            preprocessing.set_token(self.group_token)
            preprocessing_run = preprocessing.run(
                data=data,
                group_token=self.group_token,
                wait_complete=True,
            )
            if preprocessing_run.get_status() != ModelExecutionState.Succeeded:
                logger.error(f"Fail preprocessing {data} file")
                raise PreprocessingError(
                    "Fail during preprocessing script. Please check your input file and your preprocessing script."
                )
            logger.info("Data preprocessing succeeded. Downloading file...")

            preprocessing_run.download()

            logger.info("Preprocessing complete.")

            preprocessed_data_path = "./preprocessed_data.parquet"
            files = [("input", ("preprocessed_data.parquet", open(preprocessed_data_path, "rb")))]
        else:
            files = [("input", (data.split("/")[-1], open(data, "rb")))]

        logger.info("Running data prediction...")

        response = make_request(
            url=f"{self.base_url}/model/async/run/{self.group}/{self.model_hash}",
            method="POST",
            files=files,
            success_code=202,
            custom_exception=ModelError,
            custom_exception_message=f"Failed to predict data for model {self.model_hash} in group {self.group}",
            specific_error_code=404,
            logger_msg=f"Failed to predict data for model {self.model_hash} in group {self.group}",
            headers={
                "Authorization": f"Bearer {self.group_token}",
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.predict.__qualname__,
            },
        ).json()

        execution_id = response["ExecutionId"]

        if wait_complete:
            self.wait_run_ready(execution_id, self.group_token)

        logger.info("Analysis complete. Predictions are now available!")

        model_execution = ModelExecution(exec_id=execution_id, model=self)
        return model_execution

    def __call__(
        self,
        data: Union[str, Tuple[str, str], List[Tuple[str, str]], MLOpsDataset, List[MLOpsDataset]],
        preprocessing: MLOpsPreprocessing = None,
        group_token=None,
        wait_complete=True,
    ):
        """
        Run the hosted model for a specific input. It will show the result of the prediction

        Parameters
        ----------
        data: str | tuple[str, str] | list[tuple[str, str]] | MLOpsDataset | None
            Data that will be used to run the model. You can upload a dataset hash as string, a tuple with file name and file path,
            a list of tuples with file name and file path, a MLOpsDataset or a list of MLOpsDataset.
            If you provide a single string, it will consider it as a dataset hash.
        preprocessing: MLOpsPreprocessing, default=None
            Class for preprocessing data.
        group_token: str, default=None
            Token of the group
        wait_complete: bool, default=True
            Wait for model to be ready and returns a MLOpsModel instance with the new model.

        Returns
        -------
        ModelExecution
            Class to handle model execution
        """
        self.predict(data, preprocessing, group_token, wait_complete)

    def get_model_execution(self, execution_id: Union[int, str]):
        """
        Get a model execution by its id

        Parameters
        ----------
         execution_id: Union[int, str]
            Execution id of a model prediction

        Returns
        -------
            MLOpsExecution: Class to handle model execution
        """
        logger.info(f"Getting model execution {execution_id}...")
        self._describe_execution(execution_id)

        run = MLOpsExecution(
            parent_id=self.model_hash,
            exec_type="AsyncModel",
            group=self.group,
            exec_id=execution_id,
            login=self.credentials[0],
            password=self.credentials[1],
            url=self.base_url,
            group_token=self.group_token,
        )
        run.get_status()

        logger.info(f"Model execution {execution_id} successfully loaded.")
        return run


class ModelExecution:
    """
    Class to manage new asynchronous model execution. For while, it is a temporary solution

    Parameters
    ----------
    exec_id: int
        Execution id for that specific training run
    model: AsyncModel
        Asynchronous model to handle data for model execution

    Raises
    ------
    AuthenticationError
        Invalid credentials
    """

    def __init__(self, exec_id: int, model: AsyncModel) -> None:
        self.exec_id = exec_id
        self.model = model

    def __repr__(self):
        return f"AsyncModel Execution - Execution ID: {self.exec_id}"

    def __str__(self):
        return f"AsyncModel Execution - Execution ID: {self.exec_id}"

    def get_status(self):
        """
        Get the status of the asynchronous model execution.

        Returns
        -------
        str
            Status of the asynchronous model execution.
        """
        status = self.model.execution_status(execution_id=self.exec_id)
        return status.name

    def wait_ready(self):
        """
        Wait for the asynchronous model execution to finish.
        """
        self.model.wait_run_ready(execution_id=self.exec_id)

    def download(
        self,
        name: Optional[str] = "predictions.zip",
        path: Optional[str] = "./",
        group_token: Optional[str] = None,
    ):
        """
        Download the asynchronous model execution.

        Parameters
        ----------
        name: Optional[str], default="predictions.zip"
            Name of the file to be downloaded
        path: Optional[str], default="./"
            Path where to save the downloaded file.
        group_token: Optional[str], default=None
            Token of the group to download the preprocessing script execution.
        """
        url = f"{self.model.base_url}/model/async/result/{self.model.group}/{self.exec_id}"
        if group_token:
            self.model.set_token(group_token)

        response = make_request(
            url=url,
            method="GET",
            success_code=200,
            headers={
                "Authorization": f"Bearer {self.model.group_token}",
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.download.__qualname__,
            },
        )

        if not name.endswith(".zip"):
            name = f"{name}.zip"

        if not path.endswith("/"):
            path = path + "/"

        with open(path + name, "wb") as f:
            f.write(response.content)

        logger.info(f"Downloaded model execution in {path}{name}")

    def execution_info(self):
        """
        Log the information about the asynchronous model execution.
        """
        response = self.model._describe_execution(execution_id=self.exec_id)
        logger.info(f"Result:\n{parse_json_to_yaml(response)}")


class MLOpsModelClient(BaseMLOpsClient):
    """
    Class for client to access MLOps and manage models

    Parameters
    ----------
    login: str
        Login for authenticating with the client. You can also use the env variable MLOPS_USER to set this
    password: str
        Password for authenticating with the client. You can also use the env variable MLOPS_PASSWORD to set this
    url: str
        URL to MLOps Server. Default value is https://neomaril.datarisk.net, use it to test your deployment first before changing to production. You can also use the env variable MLOPS_URL to set this

    Raises
    ------
    AuthenticationError
        Invalid credentials
    ServerError
        Server unavailable

    Example
    --------
    Example 1: Creation and managing a Synchronous Model

    .. code-block:: python

        from mlops_codex.model import MLOpsModelClient
        from mlops_codex.model import MLOpsModel

        def new_sync_model(client, group, data_path):
            model = client.create_model('Model Example Sync',
                                'score',
                                data_path+'app.py',
                                data_path+'model.pkl',
                                data_path+'requirements.txt',
                                data_path+'schema.json',
                                group=group,
                                operation="Sync"
                                )

            model.register_monitoring('parse',
                            'get_shap',
                            configuration_file=data_path+'configuration.json',
                            preprocess_file=data_path+'preprocess.py',
                            requirements_file=data_path+'requirements.txt'
                            )

            return model.model_hash

        client = MLOpsModelClient('123456')
        client.create_group('ex_group', 'Group for example purpose')

        data_path = './samples/syncModel/'

        model_hash = new_sync_model(client, 'ex_group', data_path)

        model_list = client.search_models()
        print(model_list)

        model = client.get_model(model_hash, 'ex_group')

        print(model.health())

        model.wait_ready()
        model.predict(model.schema)

        print(model.get_logs(routine='Run'))

    Example 2: creation and deployment of a Asynchronous Model

    .. code-block:: python

        from mlops_codex.model import MLOpsModelClient
        from mlops_codex.model import MLOpsModel

        def new_async_model(client, group, data_path):
            model = client.create_model('Teste notebook Async',
                            'score',
                            data_path+'app.py',
                            data_path+'model.pkl',
                            data_path+'requirements.txt',
                            group=group,
                            python_version='3.9',
                            operation="Async",
                            input_type='csv'
                            )

            return model.model_hash

        def run_model(client, model_hash, data_path):
            model = client.get_model(model_hash, 'ex_group')

            execution = model.predict(data_path+'input.csv')

            return execution

        client = MLOpsModelClient('123456')
        client.create_group('ex_group', 'Group for example purpose')

        data_path = './samples/asyncModel/'

        model_hash = new_async_model(client, 'ex_group', data_path)

        execution = run_model(client, model_hash, data_path)

        execution.get_status()

        execution.download_result()
    """

    def __repr__(self) -> str:
        return f'API version {self.version} - MLOpsModelClient(url="{self.base_url}", Token="{self.user_token}")'

    def __str__(self):
        return f"MLOPS {self.base_url} Model client:{self.user_token}"

    def status(self, model_hash: str, group: str):
        """
        Gets the status of the model with the hash equal to `model_hash`

        Parameters
        ----------
        model_hash: str
            Model id (hash) from the model being searched
        group: str
            Group the model is inserted

        Raises
        ------
        ModelError
            Model unavailable

        Returns
        -------
        dict
            The model status and a message if the status is 'Failed'
        """

        return _model_status(self.base_url, self.credentials, group, model_hash)

    def get_model(
        self, model_hash: str, group: str, group_token: Optional[str] = None
    ) -> MLOpsModel:
        """
        Acess a model using its id

        Parameters
        ----------
        model_hash: str
            Model id (hash) that needs to be acessed
        group: str
            Group the model was inserted
        group_token: Optional[str], default = None
            Token of the group being accessed

        Raises
        ------
        ModelError
            Model unavailable
        ServerError
            Unknown return from server

        Returns
        -------
        MLOpsModel
            A MLOpsModel instance with the model hash from `model_hash`

        Example
        -------
        >>> model.get_model(model_hash='M9c3af308c754ee7b96b2f4a273984414d40a33be90242908f9fc4aa28ba8ec4')
        """
        logger.info(f"Trying to get a model with hash {model_hash}")

        url = f"{self.base_url}/model/describe/{group}/{model_hash}"
        token = refresh_token(*self.credentials, self.base_url)
        response = make_request(
            url=url,
            method="GET",
            success_code=200,
            headers={"Authorization": f"Bearer {token}"},
        ).json()["Description"]

        logger.info("Model has been founded")

        if response["Operation"] == "Sync":
            return SyncModel(
                name=response["Name"],
                login=self.credentials[0],
                password=self.credentials[1],
                url=self.base_url,
                model_hash=model_hash,
                group=group,
                group_token=group_token,
            )

        return AsyncModel(
            name=response["Name"],
            login=self.credentials[0],
            password=self.credentials[1],
            url=self.base_url,
            model_hash=model_hash,
            group=group,
            group_token=group_token,
        )

    def search_models(
        self,
        *,
        name: Optional[str] = None,
        state: Optional[str] = None,
        group: Optional[str] = None,
        only_deployed: bool = False,
    ) -> list:
        """
        Search for models using the name of the model

        Parameters
        ----------
        name: Optional[str], default=None
            Text that it's expected to be on the model name. It runs similar to a LIKE query on SQL
        state: Optional[str], default=None
            Text that it's expected to be on the state. It runs similar to a LIKE query on SQL
        group: Optional[str], default=None
            Text that it's expected to be on the group name. It runs similar to a LIKE query on SQL
        only_deployed: Optional[bool], default=False
            If it's True, filter only models ready to be used (status == "Deployed").

        Raises
        ------
        ServerError
            Unexpected server error

        Returns
        -------
        list
            A list with the models data, it can works like a filter depending on the arguments values
        Example
        -------
        >>> client.search_models(group='ex_group', only_deployed=True)
        """
        search_parameters = " | ".join(
            [p for p in [name, state, group, only_deployed] if p is not None]
        )
        logger.info(
            f"Trying to search for models given: {search_parameters} parameters"
        )

        url = f"{self.base_url}/model/search"

        query = {}

        if name:
            query["name"] = name

        if state:
            query["state"] = state

        if group:
            query["group"] = group

        if only_deployed:
            query["state"] = "Deployed"

        token = refresh_token(*self.credentials, self.base_url)
        response = make_request(
            url=url,
            method="GET",
            success_code=200,
            params=query,
            headers={
                "Authorization": f"Bearer {token}",
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.search_models.__qualname__,
            },
        ).json()

        logger.info(f"Found {response['Count']} models")

        results = response["Results"]
        models = []
        for result in results:
            if result["Operation"] == "Sync":
                model = SyncModel(
                    name=result["Name"],
                    login=self.credentials[0],
                    password=self.credentials[1],
                    url=self.base_url,
                    model_hash=result["ModelHash"],
                    group=result["Group"],
                )
                models.append(model)
            else:
                model = AsyncModel(
                    name=result["Name"],
                    login=self.credentials[0],
                    password=self.credentials[1],
                    url=self.base_url,
                    model_hash=result["ModelHash"],
                    group=result["Group"],
                )
                models.append(model)

        logger.info(f"Returning {len(models)} models")

        return models

    def get_logs(
        self,
        *,
        model_hash,
        start: Optional[str] = None,
        end: Optional[str] = None,
        routine: Optional[str] = None,
        log_type: Optional[str] = None,
    ):
        """
        Get the logs

        Parameters
        ----------
        model_hash: str
            Model id (hash)
        start: Optional[str], default=None
            Date to start filter. At the format aaaa-mm-dd
        end: Optional[str], default=None
            Date to end filter. At the format aaaa-mm-dd
        routine: Optional[str], default=None
            Type of routine being executed, can assume values 'Host' (for deployment logs) or 'Run' (for execution logs)
        log_type: Optional[str], default=None
            Defines the type of the logs that are going to be filtered, can assume the values 'Ok', 'Error', 'Debug' or 'Warning'

        Raises
        ------
        ServerError
            Unexpected server error

        Returns
        -------
        dict
            Logs list
        """
        url = f"{self.base_url}/model/logs/{model_hash}"
        return self._logs(
            url=url,
            credentials=self.credentials,
            start=start,
            end=end,
            routine=routine,
            type=log_type,
        )

    def __upload_model(
        self,
        *,
        model_name: str,
        model_reference: str,
        source_file: str,
        model_file: str,
        requirements_file: str,
        schema: Optional[Union[str, dict]] = None,
        group: Optional[str] = None,
        extra_files: Optional[list] = None,
        env: Optional[str] = None,
        python_version: str = "3.10",
        operation: str = "Sync",
        input_type: str = None,
    ) -> str:
        """
        Upload the files to the server

        Parameters
        ----------
        model_name: str
            The name of the model, in less than 32 characters
        model_reference: str
            The name of the scoring function inside the source file
        source_file: str
            Path of the source file. The file must have a scoring function that accepts two parameters: data (data for the request body of the model) and model_path (absolute path of where the file is located)
        model_file: str
            Path of the model pkl file
        requirements_file: str
            Path of the requirement file. The package versions must be fixed eg: pandas==1.0
        schema: Union[str, dict], optional
            Path to a JSON or XML file with a sample of the input for the entrypoint function. A dict with the sample input can be sending as well
        group: Optional[str], optional
            Group the model is inserted.
        extra_files: Optional[str], optional
            A optional list with additional files paths that should be uploaded. If the scoring function refers to this file they will be on the same folder as the source file
        env: Optional[str], default=True
            Flag that choose which environment (dev, staging, production) of MLOps you are using. The default is True
        python_version: Optional[str], default='3.10'
            Python version for the model environment. Available versions are 3.8, 3.9, 3.10
        operation: Optional[str], optional
            Defines which kind of operation is being executed (Sync or Async). Default value is Sync
        input_type: str
            The type of the input file that should be 'json', 'csv', 'parquet', 'txt', 'xls', 'xlsx'

        Raises
        ------
        InputError
            Some input parameters are invalid

        Returns
        -------
        str
            The new model id (hash)
        """

        if operation == "Async" and input_type not in ["json", "csv", "parquet"]:
            raise InputError(
                "For async models the input_type must be 'json', 'csv' or 'parquet'"
            )

        if operation == "Sync" and input_type != "json":
            raise InputError("For sync models the input_type must be 'json'")

        file_extension_validation(source_file, {"py", "ipynb"})
        file_extension_validation(schema, {"csv", "parquet", "json"})
        python_version = validate_python_version(python_version)

        upload_data = [
            ("source",(source_file.split("/")[-1], open(source_file, "rb"))),
            ("model", (model_file.split("/")[-1], open(model_file, "rb"))),
            ("requirements", (requirements_file.split("/")[-1], open(requirements_file, "rb")))
        ]

        if schema:
            file_extension_validation(schema, {"csv", "parquet", "json"})
            upload_data.append(("schema", (schema.split("/")[-1], open(schema, "rb"))))

        if env:
            file_extension_validation(env, {"env"})
            upload_data.append(("env", (env.split("/")[-1], open(env, "rb"))))

        if extra_files:
            extra_data = [
                ("extra", (c.split("/")[-1], open(c, "rb"))) for c in extra_files
            ]
            upload_data += extra_data

        form_data = {
            "name": model_name,
            "model_reference": model_reference,
            "operation": operation,
            "input_type": input_type,
            "python_version": python_version,
        }

        token = refresh_token(*self.credentials, self.base_url)

        response = make_request(
            url=f"{self.base_url}/model/upload/{group}",
            method="POST",
            success_code=201,
            data=form_data,
            files=upload_data,
            headers={
                "Authorization": f"Bearer {token}",
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.__upload_model.__qualname__,
            },
        ).json()

        model_hash = response["ModelHash"]
        msg = response["Message"]
        logger.info(f"{msg} - Hash: {model_hash}")

        return model_hash

    def create_model(
        self,
        *,
        model_name: str,
        model_reference: str,
        source_file: str,
        model_file: str,
        requirements_file: str,
        group: str,
        input_type: str,
        schema: Optional[Union[str, dict]] = None,
        extra_files: Optional[list] = None,
        env: Optional[str] = None,
        python_version: str = "3.10",
        operation: Optional[str] = "Sync",
        wait_for_ready: bool = True,
    ) -> MLOpsModel:
        """
        Deploy a new model to MLOps.

        Parameters
        ----------
        model_name: str
            The name of the model, in less than 32 characters
        model_reference: str
            The name of the scoring function inside the source file
        source_file: str
            Path of the source file. The file must have a scoring function that accepts two parameters: data (data for the request body of the model) and model_path (absolute path of where the file is located)
        model_file: str
            Path of the model pkl file
        requirements_file: str
            Path of the requirements file. The packages versions must be fixed eg: pandas==1.0
        schema: Union[str, dict]
            Path to a JSON or XML file with a sample of the input for the entrypoint function. A dict with the sample input can be sending as well. Mandatory for Sync models
        group: str
            Group the model is inserted.
        extra_files: list, optional
            A optional list with additional files paths that should be uploaded. If the scoring function refer to this file they will be on the same folder as the source file
        env: str, optional
            .env file to be used in your model environment. This will be encrypted in the server.
        python_version: str, default="3.10"
            Python version for the model environment. Available versions are 3.8, 3.9, 3.10.
        operation: Optional[str], default="Sync"
            Defines which kind operation is being executed (Sync or Async)
        input_type: str
            The type of the input file that should be 'json', 'csv' or 'parquet'
        wait_for_ready: bool, optional
            Wait for model to be ready and returns a MLOpsModel instance with the new model

        Raises
        ------
        InputError
            Some input parameters is invalid

        Returns
        -------
        MLOpsModel
            Returns the new model, if wait_for_ready=True runs the deployment process synchronously. If it's False, returns nothing after sending all the data to server and runs the deployment asynchronously
        """

        logger.info(f"Creating a new model {model_name}. Validating data")

        validate_group_existence(group, self)

        if operation.title() not in ["Sync", "Async"]:
            raise InputError("operation must be either 'Sync' or 'Async'")

        operation = operation.title()

        logger.info("Building model...")
        model_hash = self.__upload_model(
            model_name=model_name,
            model_reference=model_reference,
            source_file=source_file,
            model_file=model_file,
            requirements_file=requirements_file,
            schema=schema,
            group=group,
            extra_files=extra_files,
            python_version=python_version,
            env=env,
            operation=operation,
            input_type=input_type,
        )

        builder = SyncModel if operation == "Sync" else AsyncModel
        model = builder(
            name=model_name,
            login=self.credentials[0],
            password=self.credentials[1],
            url=self.base_url,
            model_hash=model_hash,
            group=group,
        )

        model.host(operation=operation)

        if wait_for_ready:
            model.wait_ready()

        return model

    def get_model_execution(
        self, *, model_hash: str, exec_id: str, group: Optional[str] = None
    ) -> Union[MLOpsModel, None]:
        """
        Get an execution instace (Async model only).

        Parameters
        ----------
        model_hash: str
            Model id (hash)
        exec_id: str
            Execution id
        group: str, optional
            Group name, default value is None

        Returns
        -------
        MLOpsExecution
            The new execution

        Example
        -------
        >>> model.get_model_execution(model_hash=,exec_id='1')
        """

        logger.info(f"Getting execution for {model_hash}...")

        model = self.get_model(model_hash=model_hash, group=group)
        if isinstance(model, SyncModel):
            logger.info(
                f"You can not get {model.name} execution. It must be a asynchronous model."
            )
            return None
        logger.info(f"Found execution for {model_hash}...")
        return model.get_model_execution(execution_id=exec_id)
