#!/usr/bin/env python
# coding: utf-8

import json
import os
from http import HTTPStatus
from time import sleep
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from pydantic import BaseModel, Field, PrivateAttr

from mlops_codex.__model_states import ModelExecutionState, ModelState
from mlops_codex.__utils import extract_execution_number_from_string, parse_json_to_yaml
from mlops_codex.base import BaseMLOps, BaseMLOpsClient, MLOpsExecution
from mlops_codex.dataset import MLOpsDataset
from mlops_codex.exceptions import (
    AuthenticationError,
    ExecutionError,
    GroupError,
    InputError,
    PreprocessingError,
    ServerError,
)
from mlops_codex.http_request_handler import make_request, refresh_token
from mlops_codex.logger_config import get_logger
from mlops_codex.validations import validate_group_existence, validate_python_version

logger = get_logger()


class MLOpsPreprocessingAsyncV2Client(BaseMLOpsClient):
    """
    Class to operate actions in an asynchronous pre-processing.

    Parameters
    ----------
    login: str
        Login for authenticating with the client. You can also use the env variable MLOPS_USER to set this
    password: str
        Password for authenticating with the client. You can also use the env variable MLOPS_PASSWORD to set this
    url: str
        URL to MLOps Server. Default value is https://neomaril.datarisk.net, use it to test your deployment first before changing to production. You can also use the env variable MLOPS_URL to set this
    """

    def __init__(
        self, login: str = None, password: str = None, url: str = None
    ) -> None:
        super().__init__(login=login, password=password, url=url)
        self.url = f"{self.base_url}/v2/preprocessing"

    def __register(self, payload: dict, token: str, group: str) -> str:
        """
        Register a new preprocessing script to MLOps.

        Parameters
        ----------
        payload: dict
            Payload data
        token: str
            Token to authenticate with the MLOps server
        group: str
            Name of the group to register the preprocessing to

        Returns
        -------
        str
            Preprocessing script hash

        Raises
        ------
        AuthenticationError
            Raised if there is an authentication issue.
        GroupError
            Raised if group does not exist.
        ServerError
            Raised if the server encounters an issue.
        """
        url = f"{self.url}/{group}"

        response = make_request(
            url=url,
            method="POST",
            success_code=201,
            json=payload,
            custom_exception=GroupError,
            custom_exception_message=f"Failed to create preprocessing. Group {group} does not exist.",
            specific_error_code=404,
            logger_msg=f"Group {group} does not exist.",
            headers={
                "Authorization": f"Bearer {token}",
            },
        )

        preprocessing_script_hash = response.json()["PreprocessHash"]
        return preprocessing_script_hash

    def __upload_schema(
        self,
        preprocessing_script_hash: str,
        token: str,
        schema_file: Optional[Tuple[str, str]] = None,
        schema_dataset: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Upload schema to MLOps.

        Parameters
        ----------
        preprocessing_script_hash: str
            Preprocessing script hash
        token: str
            Token to authenticate with the MLOps server
        schema_file: Tuple[str, str]
            Schema to upload
        schema_dataset: str
            Dataset to upload schema to

        Returns
        -------
        Tuple[str, str]
            Dataset hash and name of generated preprocessing script dataset

        Raises
        ------
        AuthenticationError
            Raised if there is an authentication issue.
        PreprocessingError
            Raised could not find a preprocessing script hash.
        ServerError
            Raised if the server encounters an issue.
        """

        url = f"{self.url}/{preprocessing_script_hash}/schema"

        if schema_file is not None:
            name, file_path = schema_file
            upload_data = {"schema_file": open(file_path, "rb")}
            input_data = {"dataset_name": name}

            response = make_request(
                url=url,
                method="PATCH",
                data=input_data,
                files=upload_data,
                success_code=201,
                custom_exception=PreprocessingError,
                custom_exception_message=f"Failed to create preprocessing for preprocessing hash {preprocessing_script_hash}.",
                specific_error_code=404,
                logger_msg=f"Failed to upload schema. Could not find preprocessing schema for {preprocessing_script_hash} hash.",
                headers={
                    "Authorization": f"Bearer {token}",
                },
            )
        else:
            input_data = {"dataset_hash": schema_dataset}

            response = make_request(
                url=url,
                method="PATCH",
                data=input_data,
                success_code=201,
                custom_exception=PreprocessingError,
                custom_exception_message=f"Failed to create preprocessing for preprocessing hash {preprocessing_script_hash}.",
                specific_error_code=404,
                logger_msg=f"Failed to upload schema. Could not find preprocessing schema for {preprocessing_script_hash} hash.",
                headers={
                    "Authorization": f"Bearer {token}",
                },
            )

        output_dataset_hash = response.json()["DatasetHash"]
        output_dataset_name = response.json()["DatasetName"]
        return output_dataset_hash, output_dataset_name

    def __upload_script(
        self,
        preprocessing_script_hash: str,
        script_path: str,
        entrypoint: str,
        python_version: str,
        token: str,
    ) -> None:
        """
        Upload a python script to MLOps.

        Parameters
        ----------
        preprocessing_script_hash: str
            Preprocessing script hash
        script_path: str
            Path to python script
        entrypoint: str
            Entry point function name of python script
        token: str
            Token to authenticate with the MLOps server

        Raises
        ------
        AuthenticationError
            Raised if there is an authentication issue.
        PreprocessingError
            Raised could not find a preprocessing script hash.
        ServerError
            Raised if the server encounters an issue.
        """
        upload_data = {"script": open(script_path, "rb")}
        input_data = {
            "preprocess_reference": entrypoint,
            "python_version": python_version,
        }

        url = f"{self.url}/{preprocessing_script_hash}/script-file"

        _ = make_request(
            url=url,
            method="PATCH",
            data=input_data,
            files=upload_data,
            success_code=201,
            custom_exception=PreprocessingError,
            custom_exception_message=f"Failed to create preprocessing for preprocessing hash {preprocessing_script_hash}.",
            specific_error_code=404,
            logger_msg=f"Failed to upload preprocessing script. Could not find preprocessing script for {preprocessing_script_hash} hash.",
            headers={
                "Authorization": f"Bearer {token}",
            },
        )

    def __upload_requirements(
        self, preprocessing_script_hash: str, requirements_path: str, token: str
    ) -> None:
        """
        Upload requirements to MLOps.

        Parameters
        ----------
        preprocessing_script_hash: str
            Preprocessing script hash
        requirements_path: str
            Path to requirements file
        token: str
            Token to authenticate with the MLOps server

        Raises
        ------
        AuthenticationError
            Raised if there is an authentication issue.
        PreprocessingError
            Raised could not find a preprocessing script hash.
        ServerError
            Raised if the server encounters an issue.
        """
        upload_data = {"requirements": open(requirements_path, "rb")}

        url = f"{self.url}/{preprocessing_script_hash}/requirements-file"
        _ = make_request(
            url=url,
            method="PATCH",
            files=upload_data,
            success_code=201,
            custom_exception=PreprocessingError,
            custom_exception_message=f"Failed to create preprocessing for preprocessing hash {preprocessing_script_hash}.",
            specific_error_code=404,
            logger_msg=f"Failed to upload requirements. Could not find preprocessing requirements for {preprocessing_script_hash} hash.",
            headers={
                "Authorization": f"Bearer {token}",
            },
        )

    def __upload_extras(
        self, preprocessing_script_hash: str, extra_files: Tuple[str, str]
    ) -> None:
        url = f"{self.url}/{preprocessing_script_hash}/extra-file"
        token = refresh_token(*self.credentials, self.base_url)

        file_path = extra_files[1]
        file_name = extra_files[0]

        upload_data = {"extra_file": open(file_path, "rb")}
        input_data = {"extra_file_name": file_name}

        response = make_request(
            url=url,
            method="PATCH",
            data=input_data,
            files=upload_data,
            success_code=201,
            custom_exception=PreprocessingError,
            custom_exception_message=f"Failed to create preprocessing for preprocessing hash {preprocessing_script_hash}.",
            logger_msg=f"Failed to upload preprocessing for preprocessing hash {preprocessing_script_hash} hash.",
            specific_error_code=404,
            headers={
                "Authorization": f"Bearer {token}",
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.create.__qualname__,
            },
        )
        msg = response.json()["Message"]
        logger.debug(msg)

    def host(self, preprocessing_script_hash: str, token: str) -> None:
        """
        Host a preprocessing script to MLOps.

        Parameters
        ----------
        preprocessing_script_hash: str
            Preprocessing script hash
        token: str
            Token to authenticate with the MLOps server

        Raises
        ------
        AuthenticationError
            Raised if there is an authentication issue.
        PreprocessingError
            Raised could not find a preprocessing script hash.
        ServerError
            Raised if the server encounters an issue.
        """
        url = f"{self.url}/{preprocessing_script_hash}/status"
        _ = make_request(
            url=url,
            method="PATCH",
            success_code=202,
            custom_exception=PreprocessingError,
            custom_exception_message=f"Failed to create preprocessing for preprocessing hash {preprocessing_script_hash}.",
            logger_msg=f"Failed to host preprocessing for preprocessing hash {preprocessing_script_hash} hash.",
            specific_error_code=404,
            headers={
                "Authorization": f"Bearer {token}",
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.create.__qualname__,
            },
        )

    def host_status(self, preprocessing_script_hash: str):
        """
        Get the host status for a preprocessing script.

        Parameters
        ----------
        preprocessing_script_hash: str
            Preprocessing script hash

        Returns
        -------
        Tuple[ModelState, Union[str, None]]
            Host status for a preprocessing script and dataset hash (if it is not available, it will return None).

        Raises
        ------
        AuthenticationError
            Raised if there is an authentication issue.
        PreprocessingError
            Raised could not find a preprocessing script hash.
        ServerError
            Raised if the server encounters an issue.
        """
        url = f"{self.url}/{preprocessing_script_hash}/status"
        token = refresh_token(*self.credentials, self.base_url)
        response = make_request(
            url=url,
            method="GET",
            success_code=200,
            custom_exception=PreprocessingError,
            custom_exception_message=f"Failed to create preprocessing for preprocessing hash {preprocessing_script_hash}.",
            logger_msg=f"Failed get status for preprocessing hash {preprocessing_script_hash} hash.",
            specific_error_code=404,
            headers={
                "Authorization": f"Bearer {token}",
            },
        )

        status = ModelState[response.json()["Status"]]
        if status == ModelState.Deployed:
            dataset_hash = response.json()["DatasetHash"]
            return status, dataset_hash, None
        return status, None, response.json()["Message"]

    def wait(self, preprocessing_script_hash: str, token: str):
        """
        Check host status for a preprocessing script every 30 seconds.

        Parameters
        ----------
        preprocessing_script_hash: str
            Preprocessing script hash
        token: str
            Token to authenticate with the MLOps server

        Returns
        -------
        Tuple[ModelState, Union[str, None]]
            Host status for a preprocessing script and dataset hash (if it is not available, it will return None).

        Raises
        ------
        AuthenticationError
            Raised if there is an authentication issue.
        PreprocessingError
            Raised could not find a preprocessing script hash.
        ServerError
            Raised if the server encounters an issue.
        """
        status, dataset_hash, _ = self.host_status(preprocessing_script_hash)

        print("Waiting for preprocessing script to finish...", end="", flush=True)
        while status == ModelState.Building or status == ModelState.Ready:
            sleep(30)
            status, dataset_hash, _ = self.host_status(preprocessing_script_hash)
            print(".", end="", flush=True)
        print()

        if status == ModelState.Deployed:
            logger.debug("Preprocessing script finished successfully")
            return status, dataset_hash
        return status, None

    def create(
        self,
        *,
        name: str,
        group: str,
        script_path: str,
        entrypoint_function_name: str,
        requirements_path: str,
        python_version: Optional[str] = "3.9",
        schema_files_path: Optional[
            Union[Tuple[str, str], List[Tuple[str, str]]]
        ] = None,
        env_file: Optional[str] = None,
        schema_datasets: Optional[Union[str, List[str]]] = None,
        extra_files: Union[Tuple[str, str], List[Tuple[str, str]]] = None,
        wait_read: bool = False,
    ):
        """
        Create a new preprocessing script.

        Parameters
        ----------
        name: str
            Name of the new preprocessing script
        group: str
            Group of the new preprocessing script
        schema_files_path: Optional[Union[Tuple[str, str], List[Tuple[str, str]]]
            Schema files path. It must be a tuple in the format (dataset_name, dataset_file_path). If you want to upload more than a file, send a list of tuples in the format (dataset_name, dataset_file_path).
        schema_datasets: Optional[Union[str, List[str]]]
            Dataset to upload schema to
        script_path: str
            Path to the python script
        entrypoint_function_name: str
            Name of the entrypoint function in the python script
        python_version: str
            Python version for the model environment. Available versions are 3.8, 3.9, 3.10. Defaults to '3.9'
        requirements_path: str
            Path to the requirements file
        extra_files: Union[Tuple[str, str], List[Tuple[str, str]]]
            Extra files to upload to the preprocessing script. The format must be a tuple in the format (extra_file_name, extra_file_path). If you want to upload more than a file, send a list of tuples in the format (extra_file_name, extra_file_path).
        wait_read: bool
            If true, it will wait for the preprocessing script to finish before returning. Defaults to False.

        Returns
        -------
        MLOpsPreprocessingAsyncV2
            Preprocessing async version of the new preprocessing script.
        """

        schema_file_and_input_are_none = (
            schema_datasets is None and schema_files_path is None
        )
        schema_datasets_are_not_none = (
            schema_datasets is not None and schema_files_path is not None
        )

        if schema_file_and_input_are_none:
            raise InputError(
                "You must give a dataset hash or a input file! Both are None."
            )

        if schema_datasets_are_not_none:
            raise InputError(
                "You must give a dataset hash or a input file! You tried to upload both. Choose one of them."
            )

        validate_group_existence(group, self)
        python_version = validate_python_version(python_version)

        token = refresh_token(*self.credentials, self.base_url)

        payload = {
            "Name": name,
            "Operation": "Async",
        }

        preprocessing_script_hash = self.__register(payload, token, group)

        logger.info(
            f"Creating preprocessing for preprocessing hash\n Preprocessing hash = {preprocessing_script_hash}"
        )

        if isinstance(schema_files_path, list):
            for schema in schema_files_path:
                output_dataset_hash, output_dataset_name = self.__upload_schema(
                    schema_file=schema,
                    preprocessing_script_hash=preprocessing_script_hash,
                    token=token,
                )
                logger.info(
                    f"Created dataset hash {output_dataset_hash} with name {output_dataset_name}"
                )
        elif schema_files_path is not None:
            output_dataset_hash, output_dataset_name = self.__upload_schema(
                schema_file=schema_files_path,
                preprocessing_script_hash=preprocessing_script_hash,
                token=token,
            )
            logger.info(
                f"Created dataset hash {output_dataset_hash} with name {output_dataset_name}"
            )
        elif isinstance(schema_datasets, list):
            for schema_dataset in schema_datasets:
                output_dataset_hash, output_dataset_name = self.__upload_schema(
                    schema_dataset=schema_dataset,
                    preprocessing_script_hash=preprocessing_script_hash,
                    token=token,
                )
                logger.info(
                    f"Created dataset hash {output_dataset_hash} with name {output_dataset_name}"
                )
        else:
            output_dataset_hash, output_dataset_name = self.__upload_schema(
                schema_dataset=schema_datasets,
                preprocessing_script_hash=preprocessing_script_hash,
                token=token,
            )
            logger.info(
                f"Created dataset hash {output_dataset_hash} with name {output_dataset_name}"
            )

        logger.info("Schema files uploaded")

        self.__upload_script(
            preprocessing_script_hash=preprocessing_script_hash,
            script_path=script_path,
            entrypoint=entrypoint_function_name,
            python_version=python_version,
            token=token,
        )
        logger.info("Script file uploaded")

        self.__upload_requirements(
            preprocessing_script_hash=preprocessing_script_hash,
            requirements_path=requirements_path,
            token=token,
        )
        logger.info("Requirements file uploaded")

        if env_file:
            make_request(
                url=f"{self.base_url}/v2/preprocessing/{preprocessing_script_hash}/env-file",
                method='PATCH',
                success_code=201,
                files={'env': open(env_file, 'rb')},
                headers={
                    "Authorization": f"Bearer {token}",
                },
            )
            logger.info("Environment file uploaded")

        logger.info("Hosting preprocessing script")

        if extra_files is not None:
            if isinstance(extra_files, list):
                for extra_file in extra_files:
                    self.__upload_extras(
                        preprocessing_script_hash=preprocessing_script_hash,
                        extra_files=extra_file,
                    )
                    logger.info("Successfully uploaded extra files")
            elif isinstance(extra_files, tuple):
                self.__upload_extras(
                    preprocessing_script_hash=preprocessing_script_hash,
                    extra_files=extra_files,
                )
                logger.info("Successfully uploaded extra files")
            else:
                logger.error("Unsported extra file.")

        self.host(preprocessing_script_hash=preprocessing_script_hash, token=token)

        if wait_read:
            status, _ = self.wait(
                preprocessing_script_hash=preprocessing_script_hash, token=token
            )
            msg = (
                "Successfully hosted preprocessing script"
                if status == ModelState.Deployed
                else "Failed in hosting preprocessing script"
            )
            logger.info(msg)
            return preprocessing_script_hash

        logger.info(f"Building Preprocessing script {preprocessing_script_hash}")
        return preprocessing_script_hash

    # TODO: should the user has access to this endpoint? If so, I'll create a new function
    def list_preprocessing(self):
        """
        List preprocessing scripts
        """
        token = refresh_token(*self.credentials, self.base_url)

        response = make_request(
            url=self.url,
            method="GET",
            success_code=200,
            headers={
                "Authorization": f"Bearer {token}",
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.list_preprocessing.__qualname__,
            },
        )
        json_response = response.json()["Result"]
        print(parse_json_to_yaml(json_response))

    def register_execution(self, preprocessing_script_hash: str) -> int:
        """
        Register a new execution for preprocessing script

        Parameters
        ----------
        preprocessing_script_hash: str
            Preprocessing script hash

        Returns
        -------
        int
            New execution id of the preprocessing script.

        Raises
        ------
        AuthenticationError
            Raised if there is an authentication issue.
        PreprocessingError
            Raised could not find a preprocessing script hash.
        ServerError
            Raised if the server encounters an issue.
        """
        url = f"{self.url}/{preprocessing_script_hash}/execution"
        token = refresh_token(*self.credentials, self.base_url)

        response = make_request(
            url=url,
            method="POST",
            success_code=201,
            custom_exception=PreprocessingError,
            custom_exception_message=f"Failed to create preprocessing for preprocessing hash {preprocessing_script_hash}.",
            logger_msg=f"Failed get status for preprocessing hash {preprocessing_script_hash} hash.",
            specific_error_code=404,
            headers={
                "Authorization": f"Bearer {token}",
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.register_execution.__qualname__,
            },
        )
        message = response.json()["Message"]
        logger.debug(
            f"Registered execution for preprocessing hash {preprocessing_script_hash}\n Message = {message}"
        )
        exec_id = extract_execution_number_from_string(message)
        return exec_id

    def upload_input(
        self,
        preprocessing_script_hash: str,
        execution_id: int,
        data: Union[Tuple[str, str], str] = None,
    ) -> str:
        """
        Upload an input file for a preprocessing script execution

        Parameters
        ----------
        preprocessing_script_hash: str
            Preprocessing script hash
        execution_id: int
            Execution id of the preprocessing script.
        data: tuple[str, str] | str | None
            Input file path and file name. It must be a tuple of the form (input_file_name, input_file_path).

        Returns
        -------
        str
            Dataset hash of the preprocessing script execution input.

        Raises
        ------
        AuthenticationError
            Raised if there is an authentication issue.
        PreprocessingError
            Raised could not find a preprocessing script hash.
        ServerError
            Raised if the server encounters an issue.
        """

        url = f"{self.url}/{preprocessing_script_hash}/execution/{execution_id}/input"
        token = refresh_token(*self.credentials, self.base_url)

        if isinstance(data, tuple):
            name, file_path = data
            upload_data = {"input": open(file_path, "rb")}
            payload = {"dataset_name": name}

            response = make_request(
                url=url,
                method="PATCH",
                data=payload,
                files=upload_data,
                success_code=201,
                custom_exception=PreprocessingError,
                custom_exception_message=f"Failed to create preprocessing for preprocessing hash {preprocessing_script_hash}.",
                specific_error_code=404,
                logger_msg=f"Failed to upload schema. Could not find preprocessing schema for {preprocessing_script_hash} hash.",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Neomaril-Origin": "Codex",
                    "Neomaril-Method": self.upload_input.__qualname__,
                },
            )

        else:
            payload = {"dataset_hash": data}

            response = make_request(
                url=url,
                method="PATCH",
                data=payload,
                success_code=201,
                custom_exception=PreprocessingError,
                custom_exception_message=f"Failed to create preprocessing for preprocessing hash {preprocessing_script_hash}.",
                specific_error_code=404,
                logger_msg=f"Failed to upload schema. Could not find preprocessing schema for {preprocessing_script_hash} hash.",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Neomaril-Origin": "Codex",
                    "Neomaril-Method": self.upload_input.__qualname__,
                },
            )

        dataset_hash = response.json()["DatasetHash"]
        return dataset_hash

    def run(self, preprocessing_script_hash: str, execution_id: int):
        """
        Run preprocessing script execution

        Parameters
        ----------
        preprocessing_script_hash: str
            Preprocessing script hash
        execution_id: int
            Execution id of the preprocessing script.

        Raises
        ------
        AuthenticationError
            Raised if there is an authentication issue.
        PreprocessingError
            Raised could not find a preprocessing script hash.
        ServerError
            Raised if the server encounters an issue.
        """
        url = f"{self.url}/{preprocessing_script_hash}/execution/{execution_id}/run"
        token = refresh_token(*self.credentials, self.base_url)

        _ = make_request(
            url=url,
            method="PATCH",
            success_code=201,
            custom_exception=PreprocessingError,
            custom_exception_message=f"Failed to run preprocess script for execution {execution_id} - Hash: {preprocessing_script_hash}.",
            logger_msg=f"Failed to run preprocess script for execution {execution_id} - Hash: {preprocessing_script_hash}.",
            specific_error_code=404,
            headers={
                "Authorization": f"Bearer {token}",
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.run.__qualname__,
            },
        )

    def execution_status(self, preprocessing_script_hash: str, execution_id: int):
        """
        Get execution status for preprocessing script execution

        Parameters
        ----------
        preprocessing_script_hash: str
            Preprocessing script hash
        execution_id: int
            Execution id of the preprocessing script.

        Returns
        -------
        Tuple[ModelExecutionState, Union[str, None]]
            Return the status of the execution. If the execution is successful, the output dataset hash is also returned.

        Raises
        ------
        AuthenticationError
            Raised if there is an authentication issue.
        PreprocessingError
            Raised could not find a preprocessing script hash.
        ServerError
            Raised if the server encounters an issue.
        """
        url = f"{self.url}/{preprocessing_script_hash}/execution/{execution_id}/status"
        token = refresh_token(*self.credentials, self.base_url)

        response = make_request(
            url=url,
            method="GET",
            success_code=200,
            custom_exception=PreprocessingError,
            custom_exception_message=f"Failed to get status for execution {execution_id} - Hash {preprocessing_script_hash}.",
            logger_msg=f"Failed to get status for execution {execution_id} - Hash {preprocessing_script_hash}.",
            specific_error_code=404,
            headers={
                "Authorization": f"Bearer {token}",
            },
        )

        status = ModelExecutionState[response.json()["Status"]]
        if status == ModelExecutionState.Succeeded:
            dataset_hash: str = response.json()["OutputDatasetHash"]
            return status, dataset_hash
        return status, None

    def download(
        self,
        preprocessing_script_hash: str,
        execution_id: int,
        path: Optional[str] = "./",
    ):
        """
        Download preprocessing script execution

        Parameters
        ----------
        preprocessing_script_hash: str
            Preprocessing script hash
        execution_id: int
            Execution id of the preprocessing script.
        path: str
            Path to download the output of a preprocessing script execution.

        Raises
        ------
        AuthenticationError
            Raised if there is an authentication issue.
        PreprocessingError
            Raised could not find a preprocessing script hash.
        ServerError
            Raised if the server encounters an issue.
        """
        if not path.endswith("/"):
            path += "/"

        self.execution_status(preprocessing_script_hash, execution_id)

        url = f"{self.url}/{preprocessing_script_hash}/execution/{execution_id}/result"
        token = refresh_token(*self.credentials, self.base_url)

        response = make_request(
            url=url,
            method="GET",
            success_code=200,
            custom_exception=PreprocessingError,
            custom_exception_message="Preprocessing hash or execution id not found.",
            logger_msg="Preprocessing hash or execution id not found.",
            specific_error_code=404,
            headers={
                "Authorization": f"Bearer {token}",
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.download.__qualname__,
            },
        )

        filename = "preprocessed_data.parquet"
        with open(path + filename, "wb") as preprocessed_file:
            preprocessed_file.write(response.content)

        logger.debug(f"MLOps preprocessing downloaded to {path + filename}")

    def search(
        self,
        group: Optional[str] = None,
        state: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ):
        token = refresh_token(*self.credentials, self.base_url)

        query = {}

        if state:
            query["state"] = state

        if group:
            query["group"] = group

        if start:
            query["start"] = start
        if end:
            query["end"] = end

        response = make_request(
            url=self.url,
            method="GET",
            params=query,
            success_code=200,
            headers={
                "Authorization": f"Bearer {token}",
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.search.__qualname__,
            },
        )
        return response.json()["Result"]

    def describe(self, preprocessing_script_hash: str):
        token = refresh_token(*self.credentials, self.base_url)
        url = f"{self.url}/{preprocessing_script_hash}"

        response = make_request(
            url=url,
            method="GET",
            success_code=200,
            custom_exception=PreprocessingError,
            custom_exception_message=f"Preprocessing hash {preprocessing_script_hash} not found.",
            logger_msg=f"Preprocessing hash {preprocessing_script_hash} not found.",
            specific_error_code=404,
            headers={
                "Authorization": f"Bearer {token}",
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.describe.__qualname__,
            },
        )
        return response.json()

    def describe_execution(self, preprocessing_script_hash: str, execution_id: int):
        token = refresh_token(*self.credentials, self.base_url)
        url = f"{self.url}/{preprocessing_script_hash}/execution/{execution_id}"

        response = make_request(
            url=url,
            method="GET",
            success_code=200,
            custom_exception=PreprocessingError,
            custom_exception_message=f"Preprocessing hash {preprocessing_script_hash} or execution Id {execution_id} not found.",
            logger_msg=f"Preprocessing hash {preprocessing_script_hash} or execution Id {execution_id} not found.",
            specific_error_code=404,
            headers={
                "Authorization": f"Bearer {token}",
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.describe.__qualname__,
            },
        )
        return response.json()


class MLOpsPreprocessingAsyncV2(BaseModel):
    """
    Preprocessing class to represent the new preprocessing

    Parameters
    ----------
    login: str
        Login for authenticating with the client.
    password: str
        Password for authenticating with the client.
    url: str
        URL to MLOps Server. Default value is https://neomaril.datarisk.net, use it to test your deployment first before changing to production. You can also use the env variable MLOPS_URL to set this
    name: str
        Name of the preprocessing script.
    group: str
        Name of the group where the script is hosted.
    status: ModelState
        Status of the preprocessing script.
    """

    login: str = Field(exclude=True, repr=False)
    password: str = Field(exclude=True, repr=False)
    url: str = Field(exclude=True, repr=False)
    name: str
    preprocessing_hash: str
    group: str
    status: ModelState
    _preprocessing_client: MLOpsPreprocessingAsyncV2Client = PrivateAttr(
        None, init=False
    )

    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context: Any) -> None:
        if self._preprocessing_client is None:
            self._preprocessing_client = MLOpsPreprocessingAsyncV2Client(
                login=self.login,
                password=self.password,
                url=self.url,
            )

    def host(self, wait_ready: bool = False) -> None:
        """
        Host the preprocessing script in case you it is not hosted.

        Parameters
        ----------
        wait_ready: bool
            If true, it will wait for the preprocessing script to finish before returning. Defaults to False.
        """
        if self.status != ModelExecutionState.Requested:
            logger.debug(f"Skipping {self.name} because its {self.status} status")
            return

        token = refresh_token(
            *self._preprocessing_client.credentials, self._preprocessing_client.base_url
        )
        self._preprocessing_client.host(self.preprocessing_hash, token)

        if wait_ready:
            status, _ = self._preprocessing_client.wait(
                preprocessing_script_hash=self.preprocessing_hash, token=token
            )
            self.status = status
            logger.debug(f"Preprocessing script status is {status}")
            return

        self.status = ModelState.Building

    def __wait_for_execution(self, execution_id: int):
        """
        Check the execution status of the preprocessing script every 30 seconds.

        Parameters
        ----------
        execution_id: int
            Execution id of the preprocessing script.

        Returns
        -------
        Tuple[ModelExecutionState, Union[str, None]]
            Status of the execution of the preprocessing script. If the execution is successful, the output dataset hash is also returned.
        """
        status, dataset_hash = self._preprocessing_client.execution_status(
            self.preprocessing_hash, execution_id
        )

        print("Waiting for preprocessing script to finish", end="")
        while status == ModelExecutionState.Running:
            sleep(30)
            status, dataset_hash = self._preprocessing_client.execution_status(
                self.preprocessing_hash, execution_id
            )
            print(".", end="")

        if status == ModelExecutionState.Succeeded:
            logger.debug("Preprocessing script finished successfully")
            return status, dataset_hash
        logger.debug(
            f"Preprocessing script execution {self.preprocessing_hash} is other status different than Succeeded. Current status = {status}"
        )
        return status, dataset_hash

    def run(
        self,
        *,
        input_files: Optional[Union[Tuple[str, str], List[Tuple[str, str]]]] = None,
        dataset_hashes: Optional[Union[str, List[str]]] = None,
        wait_read: Optional[bool] = False,
    ):
        """
        Create a new preprocessing script execution and host it.

        Parameters
        ----------
        input_files: Union[Tuple[str, str], List[Tuple[str, str]]]
            Input file path and file name. It must be a tuple of the form (input_file_name, input_file_path). If you wish to send more than an input file, consider send a list of tuples of the form (input_file_name, input_file_path).
        dataset_hashes: Optional[Union[str, List[str]]]
            List of dataset hashes. If you have just one dataset hash, consider send a single string.
        wait_read: Optional[bool]
            If true, it will wait for the preprocessing script execution to finish before returning. Defaults to False.
        """

        input_and_dataset_hash_are_none = input_files is None and dataset_hashes is None
        input_and_dataset_hash_are_not_none = (
            input_files is not None and dataset_hashes is not None
        )

        if input_and_dataset_hash_are_none:
            raise InputError(
                "You must give a dataset hash or a input file! Both are None."
            )

        if input_and_dataset_hash_are_not_none:
            raise InputError(
                "You must give a dataset hash or a input file! You tried both."
            )

        execution_id = self._preprocessing_client.register_execution(
            self.preprocessing_hash
        )
        logger.debug(
            f"Preprocessing script execution {self.preprocessing_hash} is registered. Execution ID = {execution_id}"
        )

        if isinstance(input_files, list):
            for input_file in input_files:
                output_dataset_hash = self._preprocessing_client.upload_input(
                    preprocessing_script_hash=self.preprocessing_hash,
                    execution_id=execution_id,
                    data=input_file,
                )
                logger.debug(
                    f"Uploaded input file {input_file} - Output Hash {output_dataset_hash}"
                )
        elif input_files is not None:
            output_dataset_hash = self._preprocessing_client.upload_input(
                preprocessing_script_hash=self.preprocessing_hash,
                execution_id=execution_id,
                data=input_files,
            )
            logger.debug(
                f"Uploaded input file {input_files} - Output Hash {output_dataset_hash}"
            )

        elif isinstance(dataset_hashes, list):
            for dataset_hash in dataset_hashes:
                output_dataset_hash = self._preprocessing_client.upload_input(
                    preprocessing_script_hash=self.preprocessing_hash,
                    execution_id=execution_id,
                )
                logger.debug(
                    f"Uploaded dataset {dataset_hash} - Output Hash {output_dataset_hash}"
                )
        else:
            output_dataset_hash = self._preprocessing_client.upload_input(
                preprocessing_script_hash=self.preprocessing_hash,
                execution_id=execution_id,
            )
            logger.debug(
                f"Uploaded dataset {dataset_hashes} - Output Hash {output_dataset_hash}"
            )

        self._preprocessing_client.run(
            preprocessing_script_hash=self.preprocessing_hash, execution_id=execution_id
        )
        logger.debug(
            f"Started preprocessing script execution {execution_id} - Hash {self.preprocessing_hash}"
        )

        if wait_read:
            self.__wait_for_execution(execution_id)
            logger.debug(
                "Script finished successfully. Consider downloading the results using the 'download()' method."
            )

    def get_execution_status(self, execution_id: int):
        """
        Get the status of the preprocessing script execution.

        Parameters
        ----------
        execution_id: int
            Execution id of the preprocessing script.
        """
        status, _ = self._preprocessing_client.execution_status(
            self.preprocessing_hash, execution_id
        )
        logger.debug(f"Status of {execution_id}: {status}")

    def download(self, execution_id: int, path: Optional[str] = "./"):
        """
        Download the preprocessing script execution.

        Parameters
        ----------
        execution_id: int
            Execution id of the preprocessing script.
        path: str, optional
            Path where to save the downloaded file.
        """
        self._preprocessing_client.download(
            preprocessing_script_hash=self.preprocessing_hash,
            execution_id=execution_id,
            path=path,
        )


class PreprocessExecution:
    """
    Class to manage new processing script executions. For while, it is a temporary solution

    Parameters
    ----------
    preprocess_hash: str
        Training id (hash) from the experiment you want to access
    group: str
        Group the training is inserted.
    exec_id: int
        Execution id for that specific training run
    login: str
        Login for authenticating with the client. You can also use the env variable MLOPS_USER to set this
    password: str
        Password for authenticating with the client. You can also use the env variable MLOPS_PASSWORD to set this
    url: str

    Raises
    ------
    TrainingError
        When the training can't be accessed in the server
    AuthenticationError
        Invalid credentials

    """

    def __init__(
        self,
        *,
        preprocess_hash: str,
        group: str,
        exec_id: int,
        login: Optional[str] = None,
        password: Optional[str] = None,
        url: str = None,
    ) -> None:
        self.preprocessing_hash = preprocess_hash
        self.group = group
        self.exec_id = exec_id
        self.__client = MLOpsPreprocessingAsyncV2Client(
            login=login,
            password=password,
            url=url,
        )

    def get_status(self):
        """
        Get the status of the preprocessing script execution.

        Returns
        -------
        str
            Status of the preprocessing script execution.
        """
        status, _ = self.__client.execution_status(
            preprocessing_script_hash=self.preprocessing_hash, execution_id=self.exec_id
        )
        return status.name

    def wait_ready(self):
        """
        Wait for the preprocessing script execution to finish.
        """
        status, _ = self.__client.execution_status(
            preprocessing_script_hash=self.preprocessing_hash, execution_id=self.exec_id
        )
        while status in [ModelExecutionState.Running, ModelExecutionState.Requested]:
            sleep(30)
            status, _ = self.__client.execution_status(
                preprocessing_script_hash=self.preprocessing_hash,
                execution_id=self.exec_id,
            )
        logger.info(
            f"Preprocessing script execution {self.preprocessing_hash} is {status.name}."
        )

    def download(self, path: Optional[str] = "./"):
        """
        Download the preprocessing script execution.

        Parameters
        ----------
        path: Optional[str]
            Path where to save the downloaded file.
        """
        self.__client.download(
            preprocessing_script_hash=self.preprocessing_hash,
            execution_id=self.exec_id,
            path=path,
        )

    def execution_info(self, generate_output=False):
        """
        Log the information about a preprocessing script execution.

        Parameters
        ----------
        generate_output: bool
            If true, output will be generated to be manipulated
        """
        response = self.__client.describe_execution(
            preprocessing_script_hash=self.preprocessing_hash, execution_id=self.exec_id
        )

        if generate_output:
            return response

        logger.info(f"Result:\n{parse_json_to_yaml(response)}")


class MLOpsPreprocessing(BaseMLOps):
    """
    Class to manage Preprocessing scripts deployed inside MLOps

    Parameters
    ----------
    login: str
        Login for authenticating with the client. You can also use the env variable MLOPS_USER to set this
    password: str
        Password for authenticating with the client. You can also use the env variable MLOPS_PASSWORD to set this
    preprocessing_id: str
        Preprocessing script id (hash) from the script you want to access
    group: str
        Group the model is inserted.
    base_url: str
        URL to MLOps Server. Default value is https://neomaril.datarisk.net/, use it to test your deployment first before changing to production. You can also use the env variable MLOPS_URL to set this

    Example
    --------
    Getting a model, testing its healthy and putting it to run the prediction

    .. code-block:: python

        from mlops_codex.preprocessing import MLOpsPreprocessingClient
        from mlops_codex.model import MLOpsModelClient

        client = MLOpsPreprocessingClient('123456')

        client.search_preprocessing()

        preprocessing = client.get_preprocessing(preprocessing_id='S72110d87c2a4341a7ef0a0cb35e483699db1df6c5d2450f92573c093c65b062', group='ex_group')

    """

    def __init__(
        self,
        *,
        preprocessing_id: str,
        login: Optional[str] = None,
        password: Optional[str] = None,
        group: Optional[str] = None,
        group_token: Optional[str] = None,
        url: Optional[str] = None,
    ) -> None:
        super().__init__(login=login, password=password, url=url)
        self.preprocessing_id = preprocessing_id
        self.group = group
        self.__token = group_token if group_token else os.getenv("MLOPS_GROUP_TOKEN")
        self.__new_preprocess_client = MLOpsPreprocessingAsyncV2Client(
            login=login, password=password, url=url
        )

        try:
            response = self.__new_preprocess_client.describe(
                preprocessing_script_hash=preprocessing_id
            )
            self.operation = "async"
            self.status = response["State"]
            self.__preprocessing_ready = self.status == "Deployed"
        except Exception:
            url = f"{self.base_url}/preprocessing/describe/{group}/{preprocessing_id}"
            response = requests.get(
                url,
                headers={
                    "Authorization": "Bearer "
                    + refresh_token(*self.credentials, self.base_url)
                },
            )
            result = response.json()["Description"]
            self.operation = result.get("Operation").lower()

            response = self.__get_status()
            self.status = response.get("Status")

            self.__preprocessing_ready = self.status == "Deployed"

    def __repr__(self) -> str:
        return f"""MLOpsPreprocessing, group="{self.group}", 
                                status="{self.status}",
                                preprocessing_id="{self.preprocessing_id}",
                                operation="{self.operation.title()}",
                                )"""

    def __str__(self):
        return (
            f'MLOPS preprocessing (Group: {self.group}, Id: {self.preprocessing_id})"'
        )

    def wait_ready(self):
        """
        Waits the pre-processing to be with status 'Deployed'

        Example
        -------
        >>> preprocessing.wait_ready()
        """
        if self.status in ["Ready", "Building"]:
            self.status = self.__get_status()["Status"]
            while self.status == "Building":
                sleep(30)
                self.status = self.__get_status()["Status"]

    def get_logs(
        self,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        routine: Optional[str] = None,
        type: Optional[str] = None,
    ):
        """
        Get the logs

        Parameters
        -----------
        start: Optional[str], optional
            Date to start filter. At the format aaaa-mm-dd
        end: Optional[str], optional
            Date to end filter. At the format aaaa-mm-dd
        routine: Optional[str], optional
            Type of routine beeing executed, can assume values Host or Run
        type: Optional[str], optional
            Defines the type of the logs that are going to be filtered, can assume the values Ok, Error, Debug or Warning

        Raises
        ------
        ServerError
            Unexpected server error

        Returns
        -------
        dict
            Logs list

        Example
        -------
        >>> preprocessing.get_logs(model_hash=,start='2023-01-31',end='2023-02-24',routine='Run')
         {'Results':
            [{'Hash': 'M9c3af308c754ee7b96b2f4a273984414d40a33be90242908f9fc4aa28ba8ec4',
                'RegisteredAt': '2023-01-31T16:06:45.5955220Z',
                'OutputType': 'Error',
                'OutputData': '',
                'Routine': 'Run'}]
         }
        """
        url = f"{self.base_url}/preprocessing/logs/{self.group}/{self.preprocessing_id}"
        return self._logs(
            url=url,
            credentials=self.credentials,
            start=start,
            end=end,
            routine=routine,
            type=type,
        )

    def set_token(self, group_token: str) -> None:
        """
        Saves the group token for this preprocessing instance.

        Parameters
        ----------
        group_token: str
            Token for executing the preprocessing (show when creating a group). You can set this using the MLOPS_GROUP_TOKEN env variable

        Example
        -------
        >>> preprocessing.set_token('6cb64889a45a45ea8749881e30c136df')
        """

        self.__token = group_token
        logger.info(f"Token for group {self.group} added.")

    def run(
        self,
        *,
        data: Union[
            str,
            Tuple[str, str],
            MLOpsDataset,
            List[Tuple[str, str]],
            List[MLOpsDataset],
        ],
        group_token: Optional[str] = None,
        wait_complete: Optional[bool] = False,
    ):
        """
        Runs a prediction from the current preprocessing.

        Parameters
        ----------
        data: str | tuple[str, str] | MLOpsDataset | list[tuple[str, str]] | list[MLOpsDataset]
            The same data that is used in the source file.
            If Sync is a dict, the keys that are needed inside this dict are the ones in the `schema` attribute.
            If Async is a string with the file path with the same filename used in the source file.
            If you wish to send more than an input file, consider send a list of tuples of the form (input_file_name, input_file_path).
        group_token: Optional[str], optional
            Token for executing the preprocessing (show when creating a group). It can be informed when getting the preprocessing or when running predictions, or using the env variable MLOPS_GROUP_TOKEN
        wait_complete: Optional[bool], optional
            Boolean that informs if a preprocessing training is completed (True) or not (False). Default value is False

        Raises
        ------
        PreprocessingError
            Pre processing is not available

        Returns
        -------
        Union[dict, MLOpsExecution, PreprocessExecution]
            The return of the scoring function in the source file for Sync preprocessing or the execution class for Async preprocessing.
        """
        try:
            execution_id = self.__new_preprocess_client.register_execution(
                preprocessing_script_hash=self.preprocessing_id
            )
            logger.info(f"Registered Preprocessing for Execution ID: {execution_id}")

            if isinstance(data, str) or isinstance(data, tuple):
                self.__new_preprocess_client.upload_input(
                    self.preprocessing_id, execution_id, data
                )
            elif isinstance(data, MLOpsDataset):
                self.__new_preprocess_client.upload_input(
                    self.preprocessing_id, execution_id, data.hash
                )
            else:
                if isinstance(data[0], MLOpsDataset):
                    data = [d.hash for d in data]
                for d in data:
                    output_dataset_hash = self.__new_preprocess_client.upload_input(
                        preprocessing_script_hash=self.preprocessing_id,
                        execution_id=execution_id,
                        data=d,
                    )
                    logger.info(
                        f"Uploaded input file {d} - Output Hash {output_dataset_hash}"
                    )

            self.__new_preprocess_client.run(
                preprocessing_script_hash=self.preprocessing_id,
                execution_id=execution_id,
            )
            logger.info(f"Preprocessing has started. Execution ID: {execution_id}")
            if wait_complete:
                status, _ = self.__new_preprocess_client.execution_status(
                    self.preprocessing_id, execution_id
                )
                print("Waiting for preprocessing script to finish...", end="")
                while status == ModelExecutionState.Running:
                    sleep(10)
                    status, dataset_hash = (
                        self.__new_preprocess_client.execution_status(
                            self.preprocessing_id, execution_id
                        )
                    )
                    print(".", end="")
                print()

                logger.info(
                    f"Preprocessing script execution finished successfully {self.preprocessing_id} status = {status}"
                )
            run = PreprocessExecution(
                preprocess_hash=self.preprocessing_id,
                group=self.group,
                exec_id=execution_id,
                login=self.credentials[0],
                password=self.credentials[1],
                url=self.base_url,
            )
            return run
        except:  # noqa: E722
            if self.__preprocessing_ready:
                if (group_token is not None) | (self.__token is not None):
                    url = f"{self.base_url}/preprocessing/{self.operation}/run/{self.group}/{self.preprocessing_id}"
                    if self.__token and not group_token:
                        group_token = self.__token
                    if group_token and not self.__token:
                        self.__token = group_token
                    if self.operation == "sync":
                        preprocessing_input = {"Input": data}

                        req = requests.post(
                            url,
                            data=json.dumps(preprocessing_input),
                            headers={
                                "Authorization": "Bearer " + group_token,
                                "Neomaril-Origin": "Codex",
                                "Neomaril-Method": self.run.__qualname__,
                            },
                        )

                        return req.json()

                    elif self.operation == "async":
                        files = {
                            "dataset": open(data, "rb"),
                        }

                        req = requests.post(
                            url,
                            files=files,
                            headers={
                                "Authorization": "Bearer " + group_token,
                                "Neomaril-Origin": "Codex",
                                "Neomaril-Method": self.run.__qualname__,
                            },
                        )

                        # TODO: Shouldn't both sync and async preprocessing have the same succeeded status code?
                        if req.status_code == 202 or req.status_code == 200:
                            message = req.json()
                            logger.info(message["Message"])
                            exec_id = message["ExecutionId"]
                            run = MLOpsExecution(
                                parent_id=self.preprocessing_id,
                                exec_type="AsyncPreprocessing",
                                exec_id=exec_id,
                                login=self.credentials[0],
                                password=self.credentials[1],
                                url=self.base_url,
                                group=self.group,
                                group_token=group_token,
                            )
                            response = run.get_status()
                            status = response["Status"]
                            if wait_complete:
                                print("Waiting the training run.", end="")
                                while status in ["Running", "Requested"]:
                                    sleep(30)
                                    print(".", end="", flush=True)
                                    response = run.get_status()
                                    status = response["Status"]
                            if status == "Failed":
                                formatted_msg = parse_json_to_yaml(response.json())
                                logger.error(
                                    f"Something went wrong...\n{formatted_msg}"
                                )
                                raise ExecutionError("Training execution failed")
                        else:
                            raise ServerError(req.text)

                else:
                    logger.error(
                        "Login or password are invalid, please check your credentials."
                    )
                    raise GroupError("Group token not informed.")

            return run  # This is a potential error

    def get_preprocessing_execution(self, exec_id: str):
        """
        Get an execution instance for that preprocessing.

        Parameters
        ----------
        exec_id: str
            Execution id

        Raises
        ------
        PreprocessingError
            If the user tries to get an execution from a Sync preprocessing

        Returns
        -------
        MlopsExecution
            An execution instance for the preprocessing.

        Example
        -------
        >>> preprocessing.get_preprocessing_execution('1')
        """
        if self.operation == "async":
            return PreprocessExecution(
                preprocess_hash=self.preprocessing_id,
                group=self.group,
                exec_id=int(exec_id),
                login=self.credentials[0],
                password=self.credentials[1],
                url=self.base_url,
            )
        raise PreprocessingError("Sync pre processing don't have executions")

    def __get_status(self):
        """
        Gets the status of the preprocessing.

        Raises
        -------
        PreprocessingError
            Execution unavailable

        Returns
        -------
        str
            The preprocessing status

        """
        try:
            status, _, message = self.__new_preprocess_client.host_status(
                preprocessing_script_hash=self.preprocessing_id
            )
            return {"Status": status.name, "Message": message}
        except:  # noqa: E722
            url = f"{self.base_url}/preprocessing/status/{self.group}/{self.preprocessing_id}"
            response = requests.get(
                url,
                headers={
                    "Authorization": "Bearer "
                    + refresh_token(*self.credentials, self.base_url)
                },
            )
            if response.status_code == 200:
                return response.json()

            formatted_msg = parse_json_to_yaml(response.json())
            logger.error(f"Something went wrong...\n{formatted_msg}")
            raise PreprocessingError("Preprocessing has failed")

    def get_datasets(self):
        token = refresh_token(*self.credentials, self.base_url)

        response = make_request(
            url=f"{self.base_url}/v2/preprocessing",
            method="GET",
            success_code=200,
            headers={
                "Authorization": f"Bearer {token}",
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.get_datasets.__qualname__,
            },
        )
        results = response.json()["Result"]
        datasets = [
            MLOpsDataset(
                login=self.credentials,
                password=self.credentials[1],
                base_url=self.base_url,
                hash=dataset["DatasetHash"],
                dataset_name=dataset["DatasetName"],
                group=result["ScriptGroupName"],
            )
            for result in results
            for dataset in result["UploadedDatasets"] + result["GeneratedDatasets"]
            if result["ScriptHash"] == self.preprocessing_id
        ]
        return datasets


class MLOpsPreprocessingClient(BaseMLOpsClient):
    """
    Class for client to access MLOps and manage Preprocessing scripts

    Parameters
    ----------
    login: str
        Login for authenticating with the client. You can also use the env variable MLOPS_USER to set this
    password: str
        Password for authenticating with the client. You can also use the env variable MLOPS_PASSWORD to set this
    url: str
        URL to MLOps Server. Default value is https://neomaril.datarisk.net/, use it to test your deployment first before changing to production. You can also use the env variable MLOPS_URL to set this

    Raises
    ------
    AuthenticationError
        Invalid credentials
    ServerError
        Server unavailable

    Example
    --------
    Example 1: Creation and managing a Synchronous Preprocess script

    .. code-block:: python

        from mlops_codex.preprocessing import MLOpsPreprocessingClient
        from mlops_codex.model import MLOpsModelClient

        client = MLOpsPreprocessingClient('123456')
        PATH = './samples/syncPreprocessing/'

        sync_preprocessing = client.create('Teste preprocessing Sync', # model_name
                            'process', # name of the scoring function
                            PATH+'app.py', # Path of the source file
                            PATH+'requirements.txt', # Path of the requirements file,
                            schema=PATH+'schema.json', # Path of the schema file, but it could be a dict (only required for Sync models)
                            # env=PATH+'.env'  #  File for env variables (this will be encrypted in the server)
                            # extra_files=[PATH+'utils.py'], # List with extra files paths that should be uploaded along (they will be all in the same folder)
                            python_version='3.9', # Can be 3.8 to 3.10
                            operation="Sync", # Can be Sync or Async
                            group='datarisk' # Model group (create one using the client)
                            )

        sync_preprocessing.set_token('TOKEN')

        result = sync_preprocessing.run({'variable': 100})
        result

    Example 2: creation and deployment of an Asynchronous Preprocess script

    .. code-block:: python

        from mlops_codex.preprocessing import MLOpsPreprocessingClient
        from mlops_codex.model import MLOpsModelClient

        client = MLOpsPreprocessingClient('123456')
        PATH = './samples/asyncPreprocessing/'

        async_preprocessing = client.create('Teste preprocessing Async', # model_name
                            'process', # name of the scoring function
                            PATH+'app.py', # Path of the source file
                            PATH+'requirements.txt', # Path of the requirements file,
                            # env=PATH+'.env',  #  File for env variables (this will be encrypted in the server)
                            # extra_files=[PATH+'input.csv'], # List with extra files paths that should be uploaded along (they will be all in the same folder)
                            python_version='3.9', # Can be 3.8 to 3.10
                            operation="Async", # Can be Sync or Async
                            group='datarisk', # Model group (create one using the client)
                            input_type='csv'
                            )

        async_preprocessing.set_token('TOKEN')

        execution = async_preprocessing.run(PATH+'input.csv')

        execution.get_status()

        execution.wait_ready()

        execution.download_result()

    Example 3: Using preprocessing with a Synchronous model

    .. code-block:: python

        from mlops_codex.preprocessing import MLOpsPreprocessingClient
        from mlops_codex.model import MLOpsModelClient

        # the sync preprocess script configuration presented before
        # ...

        model_client = MLOpsModelClient('123456')

        sync_model = model_client.get_model(group='datarisk', model_hash='M3aa182ff161478a97f4d3b2dc0e9b064d5a9e7330174daeb302e01586b9654c')

        sync_model.predict(data=sync_model.schema, preprocessing=sync_preprocessing)

    Example 4: Using preprocessing with an Asynchronous model

    .. code-block:: python

        from mlops_codex.preprocessing import MLOpsPreprocessingClient
        from mlops_codex.model import MLOpsModelClient

        # the async preprocess script configuration presented before
        # ...

        async_model = model_client.get_model(group='datarisk', model_hash='Maa3449c7f474567b6556614a12039d8bfdad0117fec47b2a4e03fcca90b7e7c')

        PATH = './samples/asyncModel/'

        execution = async_model.predict(PATH+'input.csv', preprocessing=async_preprocessing)
        execution.wait_ready()

        execution.download_result()
    """

    def __init__(
        self,
        login: Optional[str] = None,
        password: Optional[str] = None,
        url: Optional[str] = None,
    ):
        super().__init__(login=login, password=password, url=url)
        self.__new_preprocessing_client = MLOpsPreprocessingAsyncV2Client(
            login=login, password=password, url=url
        )

    def __get_preprocessing_status(self, *, preprocessing_id: str, group: str) -> dict:
        """
        Gets the status of the preprocessing with the hash equal to `preprocessing_id`

        Parameters
        ----------
        group: str
            Group the preprocessing is inserted
        preprocessing_id: str
            Pre processing id (hash) from the preprocessing being searched

        Raises
        ------
        PreprocessingError
            Pre processing unavailable

        Returns
        -------
        dict
            The preprocessing status and a message if the status is 'Failed'
        """
        try:
            status, _, message = self.__new_preprocessing_client.host_status(
                preprocessing_script_hash=preprocessing_id
            )
            return {"Status": status.name, "Message": message}
        except:  # noqa: E722
            url = f"{self.base_url}/preprocessing/status/{group}/{preprocessing_id}"
            response = requests.get(
                url=url,
                headers={
                    "Authorization": "Bearer "
                    + refresh_token(*self.credentials, self.base_url)
                },
                timeout=60,
            )

            if response.status_code not in [200, 410]:
                formatted_msg = parse_json_to_yaml(response.json())
                logger.error(f"Something went wrong...\n{formatted_msg}")
                raise PreprocessingError(
                    f'Preprocessing "{preprocessing_id}" not found'
                )

            return response.json()

    def get_preprocessing(
        self,
        *,
        preprocessing_id: str,
        group: str,
        group_token: Optional[str] = None,
        wait_complete: Optional[bool] = True,
    ) -> MLOpsPreprocessing:
        """
        Access a preprocessing using its id

        Parameters
        ----------
        preprocessing_id: str
            Pre processing id (hash) that needs to be accessed.
        group: str
            Group the preprocessing is inserted.
        group_token: Optional[str], optional
            Token for executing the preprocessing (show when creating a group). It can be informed when getting the preprocessing or when running predictions, or using the env variable MLOPS_GROUP_TOKEN
        wait_complete: Optional[bool], optional
            If the preprocessing is being deployed, wait for it to be ready instead of failing the request. Defaults to True.

        Raises
        ------
        PreprocessingError
            Pre processing unavailable
        ServerError
            Unknown return from server

        Returns
        -------
        MLOpsPreprocessing
            A MLOpsPreprocessing instance with the preprocessing hash from `preprocessing_id`
        """
        try:
            response = self.__get_preprocessing_status(
                preprocessing_id=preprocessing_id, group=group
            )
        except KeyError:
            raise PreprocessingError("Preprocessing not found")

        status = response["Status"]

        if status == "Building":
            if wait_complete:
                print("Waiting for deploy to be ready.", end="")
                while status == "Building":
                    response = self.__get_preprocessing_status(
                        preprocessing_id=preprocessing_id, group=group
                    )
                    status = response["Status"]
                    print(".", end="", flush=True)
                    sleep(10)
                print()
            else:
                logger.info("Returning preprocessing, but preprocessing is not ready.")
                MLOpsPreprocessing(
                    preprocessing_id=preprocessing_id,
                    login=self.credentials[0],
                    password=self.credentials[1],
                    group=group,
                    url=self.base_url,
                    group_token=group_token,
                )

        if status in ["Disabled", "Ready"]:
            raise PreprocessingError(
                f'Preprocessing "{preprocessing_id}" unavailable (disabled or deploy process is incomplete)'
            )
        elif status == "Failed":
            logger.error(str(response["Message"]))
            raise PreprocessingError(
                f'Preprocessing "{preprocessing_id}" deploy failed, so preprocessing is unavailable.'
            )
        elif status == "Deployed":
            logger.info(
                f"Preprocessing {preprocessing_id} its deployed. Fetching preprocessing."
            )
            return MLOpsPreprocessing(
                preprocessing_id=preprocessing_id,
                login=self.credentials[0],
                password=self.credentials[1],
                group=group,
                url=self.base_url,
                group_token=group_token,
            )
        else:
            raise ServerError("Unknown preprocessing status: ", status)

    def search_preprocessing(
        self,
        *,
        name: Optional[str] = None,
        state: Optional[str] = None,
        group: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        only_deployed: bool = False,
    ) -> list:
        """
        Search for preprocessing using the name of the preprocessing

        Parameters
        ----------
        name: Optional[str]
            Text that it's expected to be on the preprocessing name. It runs similar to a LIKE query on SQL
        state: Optional[str]
            Text that it's expected to be on the state. It runs similar to a LIKE query on SQL
        group: Optional[str]
            Text that it's expected to be on the group name. It runs similar to a LIKE query on SQL
        start: Optional[str]
            Start date to filter search record
        end: Optional[str]
            End date to filter search record
        only_deployed: Optional[bool]
            If it's True, filter only preprocessing ready to be used (status == "Deployed"). Defaults to False

        Raises
        ------
        ServerError
            Unexpected server error

        Returns
        -------
        list
            A list with the preprocessing data, it can works like a filter depending on the arguments values
        Example
        -------
        >>> client.search_preprocessing(group='ex_group', only_deployed=True)
        """
        try:
            return self.__new_preprocessing_client.search(
                group=group,
                state=state,
                start=start,
                end=end,
            )
        except:  # noqa: E722
            url = f"{self.base_url}/preprocessing/search"

            query = {}

            if name:
                query["name"] = name

            if state:
                query["state"] = state

            if group:
                query["group"] = group

            if only_deployed:
                query["state"] = "Deployed"

            if start:
                query["start"] = start
            if end:
                query["end"] = end

            response = requests.get(
                url,
                params=query,
                headers={
                    "Authorization": "Bearer "
                    + refresh_token(*self.credentials, self.base_url),
                    "Neomaril-Origin": "Codex",
                    "Neomaril-Method": self.search_preprocessing.__qualname__,
                },
            )

            if response.status_code == 200:
                results = response.json()["Results"]
                return results

            formatted_msg = parse_json_to_yaml(response.json())

            if response.status_code == 401:
                logger.error(
                    "Login or password are invalid, please check your credentials."
                )
                raise AuthenticationError("Login not authorized.")

            if response.status_code >= 500:
                logger.error("Server is not available. Please, try it later.")
                raise ServerError("Server is not available!")

            logger.error(f"Something went wrong...\n{formatted_msg}")
            raise PreprocessingError("Could not search the preprocessing script")

    def get_logs(
        self,
        *,
        preprocessing_id,
        start: Optional[str] = None,
        end: Optional[str] = None,
        routine: Optional[str] = None,
        type: Optional[str] = None,
    ):
        """
        Get the logs

        Parameters
        ----------
        preprocessing_id: str
            Pre processing id (hash)
        start: Optional[str], optional
            Date to start filter. At the format aaaa-mm-dd
        end: Optional[str], optional
            Date to end filter. At the format aaaa-mm-dd
        routine: Optional[str], optional
            Type of routine being executed, can assume values 'Host' (for deployment logs) or 'Run' (for execution logs)
        type: Optional[str], optional
            Defines the type of the logs that are going to be filtered, can assume the values 'Ok', 'Error', 'Debug' or 'Warning'

        Raises
        ------
        ServerError
            Unexpected server error

        Returns
        -------
        dict
            Logs list

        Example
        -------
        >>> preprocessing.get_logs(model_hash=,routine='Run')
         {'Results':
            [{'Hash': 'B4c3af308c3e452e7b96b2f4a273984414d40a33be90242908f9fc4aa28ba8ec4',
                'RegisteredAt': '2023-02-03T16:06:45.5955220Z',
                'OutputType': 'Ok',
                'OutputData': '',
                'Routine': 'Run'}]
         }
        """
        url = f"{self.base_url}/preprocessing/logs/{preprocessing_id}"
        return self._logs(
            url=url,
            credentials=self.credentials,
            start=start,
            end=end,
            routine=routine,
            type=type,
        )

    def __upload_preprocessing(
        self,
        *,
        preprocessing_name: str,
        preprocessing_reference: str,
        source_file: str,
        requirements_file: str,
        schema: Optional[Union[str, dict]] = None,
        group: Optional[str],
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
        preprocessing_name: str
            The name of the preprocessing, in less than 32 characters
        preprocessing_reference: str
            The name of the scoring function inside the source file
        source_file: str
            Path of the source file. The file must have a scoring function that accepts two parameters: data (data for the request body of the preprocessing) and preprocessing_path (absolute path of where the file is located)
        requirements_file: str
            Path of the requirements file. The packages versions must be fixed eg: pandas==1.0
        schema: Union[str, dict], optional
            Path to a JSON or XML file with a sample of the input for the entrypoint function. A dict with the sample input can be sending as well
        group: str, optional
            Group the preprocessing is inserted. If None the server uses 'datarisk' (public group)
        extra_files: list, optional
            A optional list with additional files paths that should be uploaded. If the scoring function refer to this file they will be on the same folder as the source file
        env: str, optional
            Flag that choose which environment (dev, staging, production) of MLOps you are using. Default is True
        python_version: str, optional
            Python version for the preprocessing environment. Available versions are 3.8, 3.9, 3.10. Defaults to '3.10'
        operation: str
            Defines which kind operation is being executed (Sync or Async). Default value is Sync
        input_type: str
            The type of the input file that should be 'json', 'csv' or 'parquet'

        Raises
        ------
        InputError
            Some input parameters its invalid

        Returns
        -------
        str
            The new preprocessing id (hash)
        """
        url = f"{self.base_url}/preprocessing/register/{group}"

        file_extesions = {"py": "script.py", "ipynb": "notebook.ipynb"}

        upload_data = [
            (
                "source",
                (file_extesions[source_file.split(".")[-1]], open(source_file, "rb")),
            ),
            ("requirements", ("requirements.txt", open(requirements_file, "rb"))),
        ]

        if schema:
            if isinstance(schema, str):
                schema_file = open(schema, "rb")
            elif isinstance(schema, dict):
                schema_file = json.dumps(schema)
            upload_data.append(("schema", (schema.split("/")[-1], schema_file)))
        else:
            raise InputError(
                "Schema file is mandatory for preprocessing, choose a input type from json, parquet or csv"
            )

        if env:
            upload_data.append(("env", (".env", open(env, "r"))))

        if extra_files:
            extra_data = [
                ("extra", (c.split("/")[-1], open(c, "rb"))) for c in extra_files
            ]

            upload_data += extra_data

        form_data = {
            "name": preprocessing_name,
            "script_reference": preprocessing_reference,
            "operation": operation,
            "python_version": "Python" + python_version.replace(".", ""),
        }

        response = requests.post(
            url,
            data=form_data,
            files=upload_data,
            headers={
                "Authorization": "Bearer "
                + refresh_token(*self.credentials, self.base_url)
            },
        )

        if response.status_code == 201:
            data = response.json()
            preprocessing_id = data["Hash"]
            logger.info(
                f'{data["Message"]} - Hash: "{preprocessing_id}" with response {response.text}'
            )
            return preprocessing_id

        formatted_msg = parse_json_to_yaml(response.json())

        if response.status_code == 401:
            logger.error(
                "Login or password are invalid, please check your credentials."
            )
            raise AuthenticationError("Login not authorized.")

        if response.status_code >= 500:
            logger.error("Server is not available. Please, try it later.")
            raise ServerError("Server is not available!")

        logger.error(f"Something went wrong...\n{formatted_msg}")
        raise InputError("Invalid parameters for preprocessing creation")

    def __host_preprocessing(
        self, *, operation: str, preprocessing_id: str, group: str
    ) -> None:
        """
        Builds the preprocessing execution environment

        Parameters
        ----------
        operation: str
            The preprocessing operation type (Sync or Async)
        preprocessing_id: str
            The uploaded preprocessing id (hash)
        group: str
            Group the preprocessing is inserted. Default is 'datarisk' (public group)

        Raises
        ------
        InputError
            Some input parameters its invalid
        """

        url = (
            f"{self.base_url}/preprocessing/{operation}/host/{group}/{preprocessing_id}"
        )

        response = requests.get(
            url,
            headers={
                "Authorization": "Bearer "
                + refresh_token(*self.credentials, self.base_url),
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.create.__qualname__,
            },
        )

        if response.status_code == 202:
            logger.info(f"Preprocessing host in process - Hash: {preprocessing_id}")
            return HTTPStatus.OK

        formatted_msg = parse_json_to_yaml(response.json())

        if response.status_code == 401:
            logger.error(
                "Login or password are invalid, please check your credentials."
            )
            raise AuthenticationError("Login not authorized.")

        if response.status_code >= 500:
            logger.error("Server is not available. Please, try it later.")
            raise ServerError("Server is not available!")

        logger.error(f"Something went wrong...\n{formatted_msg}")
        raise InputError("Invalid parameters for preprocessing creation")

    def create(
        self,
        *,
        preprocessing_name: str,
        preprocessing_reference: str,
        source_file: str,
        requirements_file: str,
        group: str,
        schema: Optional[Union[str, Dict, List[Tuple[str, str]]]] = None,
        extra_files: Optional[
            Union[List, Tuple[str, str], List[Tuple[str, str]]]
        ] = None,
        env: Optional[str] = None,
        python_version: str = "3.10",
        operation="Sync",
        input_type: str = "json|csv|parquet",
        wait_complete: bool = True,
    ) -> MLOpsPreprocessing:
        """
        Deploy a new preprocessing to MLOps.

        Parameters
        ----------
        preprocessing_name: str
            The name of the preprocessing, in less than 32 characters
        preprocessing_reference: str
            The name of the scoring function inside the source file
        source_file: str
            Path of the source file. The file must have a scoring function that accepts two parameters: data (data for the request body of the preprocessing) and preprocessing_path (absolute path of where the file is located)
        requirements_file: str
            Path of the requirements file. The packages versions must be fixed eg: pandas==1.0
        group: str
            Group the preprocessing is inserted.
        schema: Optional[Union[str, Dict, List[Tuple[str, str]]]]
            Path to a JSON, XML, CSV or PARQUET file with a sample of the input for the entrypoint function. A dict with the sample input can be sending as well.
            For async models, send a parquet or csv file
            For sync models, send a json or xml file
            If you want to upload more than a file, send a list of tuples in the format (dataset_name, dataset_file_path).
        extra_files: Optional[Union[List, Tuple[str, str], List[Tuple[str, str]]]]
            A optional list with additional files paths that should be uploaded. If the scoring function refer to this file they will be on the same folder as the source file
            If you will use the extras files in the multiple preprocessing, you must upload a tuple in the format (extra_file_name, extra_file_path) or a list of tuples in that format.
        env: Optional[str]
            Flag that choose which environment (dev, staging, production) of MLOps you are using. Default is True
        python_version: Optional[str], optional
            Python version for the preprocessing environment. Available versions are 3.8, 3.9, 3.10. Defaults to '3.10'
        operation: str
            Defines which kind operation is being executed (Sync or Async). Default value is Sync
        input_type: str
            The type of the input file that should be 'json', 'csv' or 'parquet'
        wait_complete: Optional[bool]
            Wait for preprocessing to be ready and returns a MLOpsPreprocessing instance with the new preprocessing. Defaults to True

        Raises
        ------
        InputError
            Some input parameters its invalid

        Returns
        -------
        MLOpsPreprocessing
            Returns the new preprocessing, if wait_for_ready=True runs the deployment process synchronously. Otherwise, returns nothing after sending all the data to server and runs the deployment asynchronously
        """

        validate_group_existence(group, self)

        if operation == "Async":
            preprocessing_id = self.__new_preprocessing_client.create(
                name=preprocessing_name,
                group=group,
                script_path=source_file,
                entrypoint_function_name=preprocessing_reference,
                requirements_path=requirements_file,
                python_version=python_version,
                schema_files_path=schema,
                env_file=env,
                extra_files=extra_files,
                wait_read=wait_complete,
            )

            # The MLOpsPreprocessingAsyncV2Client is hosted internally
            return self.get_preprocessing(
                preprocessing_id=preprocessing_id,
                group=group,
                wait_complete=wait_complete,
            )

        preprocessing_id = self.__upload_preprocessing(
            preprocessing_name=preprocessing_name,
            preprocessing_reference=preprocessing_reference,
            source_file=source_file,
            requirements_file=requirements_file,
            schema=schema,
            group=group,
            extra_files=extra_files,
            python_version=python_version,
            env=env,
            operation=operation,
            input_type=input_type,
        )

        self.__host_preprocessing(
            operation=operation.lower(), preprocessing_id=preprocessing_id, group=group
        )

        return self.get_preprocessing(
            preprocessing_id=preprocessing_id, group=group, wait_complete=wait_complete
        )

    def get_execution(
        self, preprocessing_id: str, exec_id: str, group: Optional[str] = None
    ) -> MLOpsExecution:
        """
        Get an execution instance (Async preprocessing only).

        Parameters
        ----------
        preprocessing_id: str
            Pre processing id (hash)
        exec_id: str
            Execution id
        group: str, optional
            Group name, default value is None

        Returns
        -------
        MLOpsExecution
            The new execution
        """
        return self.get_preprocessing(
            preprocessing_id=preprocessing_id, group=group
        ).get_preprocessing_execution(exec_id)
