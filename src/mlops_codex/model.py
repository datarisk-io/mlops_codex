#!/usr/bin/env python
# coding: utf-8

import json
import os
from http import HTTPStatus
from time import sleep
from typing import Optional, Union

import requests

from mlops_codex.__model_states import ModelState
from mlops_codex.__utils import (
    parse_dict_or_file,
    parse_json_to_yaml,
)
from mlops_codex.base import BaseMLOps, BaseMLOpsClient, MLOpsExecution
from mlops_codex.datasources import MLOpsDataset
from mlops_codex.exceptions import (
    AuthenticationError,
    ExecutionError,
    InputError,
    ModelError,
    PreprocessingError,
    ServerError,
)
from mlops_codex.http_request_handler import refresh_token, try_login
from mlops_codex.logger_config import get_logger
from mlops_codex.preprocessing import MLOpsPreprocessing
from mlops_codex.validations import validate_group_existence, validate_python_version

logger = get_logger()


class MLOpsModel(BaseMLOps):
    """
    Class to manage Models deployed inside MLOps

    Parameters
    ----------
    login: str
        Login for authenticating with the client. You can also use the env variable MLOPS_USER to set this
    password: str
        Password for authenticating with the client. You can also use the env variable MLOPS_PASSWORD to set this
    model_id: str
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

    Example
    --------
    Getting a model, testing its health and running the prediction

    .. code-block:: python

        from mlops_codex.model import MLOpsModelClient
        from mlops_codex.model import MLOpsModel

        client = MLOpsModelClient('123456')

        model = client.get_model(model_id='M9c3af308c754ee7b96b2f4a273984414d40a33be90242908f9fc4aa28ba8ec4',
                                 group='ex_group')

        if model.health() = 'OK':
            model.wait_ready()
            model.predict(model.schema)
        else:
            model.restart_model(False)
            model.wait_ready()
            model.predict(model.schema)

    """

    def __init__(
        self,
        *,
        model_id: str,
        group: str,
        login: Optional[str] = None,
        password: Optional[str] = None,
        group_token: Optional[str] = None,
        url: Optional[str] = None,
    ) -> None:
        super().__init__(login=login, password=password, url=url)

        self.model_id = model_id
        self.group = group
        self.__token = group_token if group_token else os.getenv("MLOPS_GROUP_TOKEN")

        url = f"{self.base_url}/model/describe/{self.group}/{self.model_id}"
        response = requests.get(
            url,
            headers={
                "Authorization": "Bearer "
                + refresh_token(*self.credentials, self.base_url)
            },
        )

        formatted_msg = parse_json_to_yaml(response.json())

        if response.status_code == 401:
            logger.error(
                "Login or password are invalid, please check your credentials."
            )
            raise AuthenticationError("Login not authorized.")

        if response.status_code == 404:
            logger.error(f"Something went wrong...\n{formatted_msg}")
            raise ModelError(f'Model "{model_id}" not found.')

        if response.status_code >= 500:
            logger.error("Server is not available. Please, try it later.")
            raise ServerError("Server is not available!")

        self.model_data = response.json()["Description"]
        self.name = self.model_data["Name"]
        self.status = ModelState[self.model_data["Status"]]
        self.operation = self.model_data["Operation"].lower()
        self.docs = (
            f"{self.base_url}/model/{self.operation}/docs/{self.group}/{self.model_id}"
        )
        self.__model_ready = self.status == ModelState.Deployed

    def __repr__(self) -> str:
        status = self.__get_status()
        return f"""MLOpsModel(name="{self.name}", group="{self.group}", 
                                status="{status}",
                                model_id="{self.model_id}",
                                operation="{self.operation.title()}",
                                )"""

    def __str__(self):
        return f'MLOPS model "{self.name} (Group: {self.group}, Id: {self.model_id})"'

    def __get_status(self):
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
        url = f"{self.base_url}/model/status/{self.group}/{self.model_id}"
        response = requests.get(
            url,
            headers={
                "Authorization": "Bearer "
                + refresh_token(*self.credentials, self.base_url)
            },
        )
        if response.status_code == 200:
            return ModelState[response.json().get("Status")]

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
        raise ModelError("Could not get the status of the model")

    def wait_ready(self):
        """
        Waits the model to be with status 'Deployed'

        Example
        -------
        >>> model.wait_ready()
        """
        if self.status in [ModelState.Ready, ModelState.Building]:
            self.status = self.__get_status()
            while self.status == ModelState.Building:
                sleep(30)
                self.status = self.__get_status()

    def health(self) -> str:
        """
        Get the model deployment process health state.

        Raises
        ------
        AuthenticationError
            Raised if there is an authentication issue.
        ServerError
            Raised if the server encounters an issue.
        ModelError
            Raised if it can not get the health of the model

        Returns
        -------
        str
            OK - if it is possible to get the health state
            NOK - if an exception occurs

        Example
        -------
        >>> model.health()
         'OK'
        """
        if self.operation == "async":
            try:
                try_login(
                    *self.credentials,
                    self.base_url,
                )
                return "OK"
            except Exception as e:
                logger.error("Server error: " + e)
                return "NOK"
        elif self.operation == "sync":
            url = f"{self.base_url}/model/sync/health/{self.group}/{self.model_id}"
            response = requests.get(
                url,
                headers={
                    "Authorization": "Bearer " + self.__token,
                    "Neomaril-Origin": "Codex",
                    "Neomaril-Method": self.health.__qualname__,
                },
            )
            if response.status_code == 200:
                return response.json()["Message"]

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
            raise ModelError("Could not get the health of the model")

    def restart_model(self, *, wait_for_ready: bool = True):
        """
        Restart a model deployment process health state. Be sure your model is one of these states:
            - Deployed;
            - Disabled;
            - DisabledRecovery;
            - FailedRecovery.

        Parameters
        -----------
        wait_for_ready: bool
            If the model is being deployed, wait for it to be ready instead of failing the request. Defaults to True

        Raises
        ------
        AuthenticationError
            Raised if there is an authentication issue.
        ServerError
            Raised if the server encounters an issue.
        ModelError
            Raised if model could not be restarted.

        Example
        -------
        >>> model.restart_model()
        """

        url = f"{self.base_url}/model/restart/{self.group}/{self.model_id}"
        response = requests.get(
            url,
            headers={
                "Authorization": "Bearer "
                + refresh_token(*self.credentials, self.base_url),
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.restart_model.__qualname__,
            },
        )

        formatted_msg = parse_json_to_yaml(response.json())

        if response.status_code == 401:
            logger.error(
                "Login or password are invalid, please check your credentials."
            )
            raise AuthenticationError("Login not authorized.")

        if response.status_code >= 500:
            logger.error("Server is not available. Please, try it later.")
            raise ServerError("Server is not available!")

        if response.status_code != 200:
            logger.error(f"Something went wrong...\n{formatted_msg}")
            raise ModelError("Could not restart the model")

        logger.info("Model is restarting")
        self.status = self.__get_status()
        if wait_for_ready:
            print("Waiting for deploy to be ready.", end="")
            while self.status == ModelState.Building:
                sleep(30)
                self.status = self.__get_status()
                print(".", end="", flush=True)
        print("Model is deployed", flush=True)

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
        >>> model.get_logs(start='2023-01-31', end='2023-02-24', routine='Run', type='Error')
         {'Results':
            [{'ModelHash': 'M9c3af308c754ee7b96b2f4a273984414d40a33be90242908f9fc4aa28ba8ec4',
                'RegisteredAt': '2023-01-31T16:06:45.5955220Z',
                'OutputType': 'Error',
                'OutputData': '',
                'Routine': 'Run'}]
         }
        """
        url = f"{self.base_url}/model/logs/{self.group}/{self.model_id}"
        return self._logs(
            url=url,
            credentials=self.credentials,
            start=start,
            end=end,
            routine=routine,
            type=type,
        )

    def delete(self):
        """
        Deletes the current model. IMPORTANT! For now this is irreversible, if you want to use the model again later you will need to upload again (and it will have a new ID).

        Raises
        ------
        AuthenticationError
            Raised if there is an authentication issue.
        ServerError
            Model deleting failed

        Returns
        -------
        str
            If model is at status=Deployed deletes the model and return a json with his information.
            If it isn't Deployed it returns the message that the model is under another state

        Example
        -------
        >>> model.delete()
        """

        token = refresh_token(*self.credentials, self.base_url)
        req = requests.delete(
            f"{self.base_url}/model/delete/{self.group}/{self.model_id}",
            headers={
                "Authorization": "Bearer " + token,
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.delete.__qualname__,
            },
        )

        formatted_msg = parse_json_to_yaml(req.json())

        if req.status_code == 401:
            logger.error(
                "Login or password are invalid, please check your credentials."
            )
            raise AuthenticationError("Login not authorized.")

        if req.status_code >= 500:
            logger.error("Server is not available. Please, try it later.")
            raise ServerError("Server is not available!")

        if req.status_code != 200:
            logger.error(f"Something went wrong...\n{formatted_msg}")
            raise ModelError("Failed to delete model.")

        response = requests.get(
            f"{self.base_url}/model/describe/{self.group}/{self.model_id}",
            headers={"Authorization": "Bearer " + token},
        )

        self.model_data = response.json()["Description"]
        self.status = ModelState[self.model_data["Status"]]
        self.__model_ready = False

        return req.json()

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

        Returns
        -------
        dict
            status=Deployed: disables the model and return a json.
            If it isn't Deployed it returns the message that the model is under another state

        Example
        -------
        >>> model.disable()

        """

        token = refresh_token(*self.credentials, self.base_url)
        req = requests.post(
            f"{self.base_url}/model/disable/{self.group}/{self.model_id}",
            headers={
                "Authorization": "Bearer " + token,
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.disable.__qualname__,
            },
        )

        formatted_msg = parse_json_to_yaml(req.json())

        if req.status_code == 401:
            logger.error(
                "Login or password are invalid, please check your credentials."
            )
            raise AuthenticationError("Login not authorized.")

        if req.status_code >= 500:
            logger.error("Server is not available. Please, try it later.")
            raise ServerError("Server is not available!")

        if req.status_code != 200:
            logger.error(f"Something went wrong...\n{formatted_msg}")
            raise ModelError("Failed to delete model.")

        response = requests.get(
            f"{self.base_url}/model/describe/{self.group}/{self.model_id}",
            headers={"Authorization": "Bearer " + token},
        )

        self.model_data = response.json()["Description"]
        self.status = ModelState[self.model_data["Status"]]
        self.__model_ready = False

        print(f"The model {self.model_id} was disabled")

        return req.json()

    def set_token(self, group_token: str) -> None:
        """
        Saves the group token for this model instance.

        Parameters
        ----------
        group_token: str
            Token for executing the model (show when creating a group). You can set this using the MLOPS_GROUP_TOKEN env variable

        Example
        -------
        >>> model.set_token('6cb64889a45a45ea8749881e30c136df')
        """

        self.__token = group_token
        logger.info(f"Token for group {self.group} added.")

    def predict(
        self,
        *,
        data: Optional[Union[dict, str, MLOpsExecution]] = None,
        dataset: Union[str, MLOpsDataset] = None,
        preprocessing: Optional[MLOpsPreprocessing] = None,
        group_token: Optional[str] = None,
        wait_complete: Optional[bool] = False,
    ) -> Union[dict, MLOpsExecution]:
        """
        Runs a prediction from the current model.

        Parameters
        ----------
        data: Union[dict, str]
            The same data that is used in the source file.
            If Sync is a dict, the keys that are needed inside this dict are the ones in the `schema` attribute.
            If Async is a string with the file path with the same filename used in the source file.
        group_token: Optional[str], optional
            Token for executing the model (show when creating a group). It can be informed when getting the model or when running predictions, or using the env variable MLOPS_GROUP_TOKEN
        wait_complete: Optional[bool], optional
            Boolean that informs if a model training is completed (True) or not (False). Default value is False

        Raises
        ------
        ModelError
            Model is not available
        InputError
            Model requires a dataset or a data input

        Returns
        -------
        Union[dict, MLOpsExecution]
            The return of the scoring function in the source file for Sync models or the execution class for Async models.
        """
        if not (data or dataset):
            raise InputError(
                "Invalid data input. Run training requires a data or dataset"
            )
        if self.__model_ready:
            if (group_token is not None) | (self.__token is not None):
                url = f"{self.base_url}/model/{self.operation}/run/{self.group}/{self.model_id}"
                if self.__token and not group_token:
                    group_token = self.__token
                if group_token and not self.__token:
                    self.__token = group_token
                if self.operation == "sync":
                    model_input = {"Input": data}

                    if preprocessing:
                        model_input["ScriptHash"] = preprocessing.preprocessing_id

                    req = requests.post(
                        url,
                        data=json.dumps(model_input),
                        headers={
                            "Authorization": "Bearer " + group_token,
                            "Neomaril-Origin": "Codex",
                            "Neomaril-Method": self.predict.__qualname__,
                        },
                    )

                    return req.json()

                elif self.operation == "async":
                    if preprocessing:
                        if preprocessing.operation == "async":
                            preprocessing.set_token(group_token)
                            pre_run = preprocessing.run(data=data)
                            pre_run.wait_ready()
                            if pre_run.status != "Succeeded":
                                logger.error(
                                    "Preprocessing failed, we wont send any data to it"
                                )
                                logger.info("Returning Preprocessing run instead.")
                                return pre_run
                            data = "./result_preprocessing"
                            pre_run.download_result(
                                path="./", filename="result_preprocessing"
                            )
                        else:
                            raise PreprocessingError(
                                "Can only use async preprocessing with async models"
                            )

                    form_data = {}
                    if data:
                        files = [("input", (data.split("/")[-1], open(data, "rb")))]
                    elif dataset:
                        dataset_hash = (
                            dataset
                            if isinstance(dataset, str)
                            else dataset.dataset_hash
                        )
                        form_data["dataset_hash"] = dataset_hash

                    req = requests.post(
                        url,
                        files=files,
                        data=form_data,
                        headers={
                            "Authorization": "Bearer " + group_token,
                            "Neomaril-Origin": "Codex",
                            "Neomaril-Method": self.predict.__qualname__,
                        },
                    )

                    if req.status_code == 202:
                        message = req.json()
                        logger.info(message["Message"])
                        exec_id = message["ExecutionId"]
                        run = MLOpsExecution(
                            parent_id=self.model_id,
                            exec_type="AsyncModel",
                            group=self.group,
                            exec_id=exec_id,
                            login=self.credentials[0],
                            password=self.credentials[1],
                            url=self.base_url,
                            group_token=group_token,
                        )
                        response = run.get_status()
                        status = response["Status"]
                        if wait_complete:
                            run.wait_ready()
                        if status == "Failed":
                            logger.error(response["Message"])
                            raise ExecutionError("Training execution failed")
                        return run
                    elif req.status_code >= 500:
                        logger.error("Server is not available. Please, try it later.")
                        raise ServerError("Server is not available!")
                    else:
                        logger.error(req.text)
                        raise Exception("Unexpected error")

            else:
                raise InputError("Group token not informed")
        else:
            url = f"{self.base_url}/model/describe/{self.group}/{self.model_id}"
            response = requests.get(
                url,
                headers={
                    "Authorization": "Bearer "
                    + refresh_token(*self.credentials, self.base_url),
                    "Neomaril-Origin": "Codex",
                    "Neomaril-Method": self.predict.__qualname__,
                },
            ).json()["Description"]
            if response["Status"] == "Deployed":
                self.model_data = response
                self.status = ModelState[response["Status"]]
                self.__model_ready = True
                return self.predict(
                    data=data,
                    dataset=dataset,
                    preprocessing=preprocessing,
                    group_token=group_token,
                    wait_complete=wait_complete,
                )

            else:
                raise ModelError("Model is not available to predictions")

    def generate_predict_code(self, *, language: str = "curl") -> str:
        """
        Generates predict code for the model to be used outside MLOps Codex

        Parameters
        ----------
        language: str
            The generated code language. Supported languages are 'curl', 'python' or 'javascript'

        Raises
        ------
        InputError
            Unsupported language

        Returns
        -------
        str
            The generated code.
        """
        if language not in ["curl", "python", "javascript"]:
            raise InputError("Suported languages are curl, python or javascript")

        if self.operation == "sync":
            payload = json.dumps({"Input": {"DATA": "DATA"}})
            base_url = self.base_url
            if language == "curl":
                return f"""curl --request POST \\
                    --url {base_url}/model/sync/run/{self.group}/{self.model_id} \\
                    --header 'Authorization: Bearer TOKEN' \\
                    --header 'Content-Type: application/json' \\
                    --data '{payload}'
                """
            if language == "python":
                return f"""
                    import requests

                    url = "{base_url}/model/sync/run/{self.group}/{self.model_id}"

                    payload = {payload}
                    headers = {{
                        "Content-Type": "application/json",
                        "Authorization": "Bearer TOKEN"
                    }}

                    response = requests.request("POST", url, json=payload, headers=headers)

                    print(response.text)
                """
            if language == "javascript":
                return f"""
                    const options = {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json', Authorization: 'Bearer TOKEN'}},
                    body: '{payload}'
                    }};

                    fetch('{base_url}/model/sync/run/{self.group}/{self.model_id}', options)
                    .then(response => response.json())
                    .then(response => console.log(response))
                    .catch(err => console.error(err));

                """
        if self.operation == "async":
            if language == "curl":
                return f"""
                    curl --request POST \
                    --url {self.base_url}/model/async/run/{self.group}/{self.model_id} \\
                    --header 'Authorization: Bearer TOKEN' \\
                    --header 'Content-Type: multipart/form-data' \\
                    --form "input=@/path/to/file"
                """
            if language == "python":
                return f"""
                    import requests

                    url = "{self.base_url}/model/async/run/{self.group}/{self.model_id}"

                    upload_data = [
                        ("input", ('filename', open('/path/to/file', 'rb'))),
                    ]

                    headers = {{
                        "Content-Type": "multipart/form-data",
                        "Authorization": "Bearer TOKEN"
                    }}

                    response = requests.request("POST", url, files=upload_data, headers=headers)

                    print(response.text)
                """
            if language == "javascript":
                return f"""
                    const form = new FormData();
                    form.append("input", "/path/to/file");

                    const options = {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'multipart/form-data',
                        Authorization: 'Bearer TOKEN'
                    }}
                    }};

                    options.body = form;

                    fetch('{self.base_url}/model/async/run/{self.group}/{self.model_id}', options)
                    .then(response => response.json())
                    .then(response => console.log(response))
                    .catch(err => console.error(err));
                """

    def __call__(self, data: dict) -> dict:
        return self.predict(data=data)

    def get_model_execution(self, exec_id: str) -> MLOpsExecution:
        """
        Get an execution instace for that model.

        Parameters
        ----------
        exec_id: str
            Execution id

        Raises
        ------
        ModelError
            If the user tries to get an execution from a Sync model

        Example
        -------
        >>> model.get_model_execution('1')
        """
        if self.operation == "async":
            run = MLOpsExecution(
                parent_id=self.model_id,
                exec_type="AsyncModel",
                group=self.group,
                exec_id=exec_id,
                login=self.credentials[0],
                password=self.credentials[1],
                url=self.base_url,
                group_token=self.__token,
            )
            run.get_status()
            return run
        else:
            raise ModelError("Sync models don't have executions")

    def __host_monitoring_status(self, *, group: str, model_id: str, period: str):
        """
        Get the host status for the monitoring configuration

        Parameters
        ----------
        group: str
            Group the model is inserted.
        model_id: str
            The uploaded model id (hash)
        period: str
            The monitoring period (Day, Week, Month)

        Raises
        ------
        ExecutionError
            Monitoring host failed
        ServerError
            Unexpected server error
        """
        url = f"{self.base_url}/monitoring/status/{group}/{model_id}/{period}"

        response = requests.get(
            url,
            headers={
                "Authorization": "Bearer "
                + refresh_token(*self.credentials, self.base_url)
            },
        )

        if response.status_code == 200:
            message = response.json()

            status = message["Status"]
            if status == "Validating":
                logger.info("Waiting the monitoring host.")
                sleep(30)
                self.__host_monitoring_status(
                    group=group, model_id=model_id, period=period
                )  # recursive
            if status == "Validated":
                logger.info(f'Model monitoring host validated - Hash: "{model_id}"')

            if status == "Invalidated":
                res_message = message["Message"]
                logger.error(f"Model monitoring host message: {res_message}")
                raise ExecutionError("Monitoring host failed")

        if response.status_code == 401:
            logger.error(
                "Login or password are invalid, please check your credentials."
            )
            raise AuthenticationError("Login not authorized.")

        if response.status_code >= 500:
            logger.error(response.text)
            raise ModelError("Could not get host monitoring status")

    def __host_monitoring(self, *, group: str, model_id: str, period: str):
        """
        Host the monitoring configuration

        Parameters
        ----------
        group: str
            Group the model is inserted.
        model_id: str
            The uploaded model id (hash)
        period: str
            The monitoring period (Day, Week, Month)

        Raises
        ------
        InputError
            Monitoring host error
        """
        url = f"{self.base_url}/monitoring/host/{group}/{model_id}/{period}"

        response = requests.get(
            url,
            headers={
                "Authorization": "Bearer "
                + refresh_token(*self.credentials, self.base_url)
            },
        )

        if response.status_code == 200:
            logger.info(f'Model monitoring host started - Hash: "{model_id}"')
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

        logger.error(f"Model monitoring host error:\n{formatted_msg}")
        raise InputError("Monitoring host error")

    def register_monitoring(
        self,
        *,
        preprocess_reference: str,
        shap_reference: str,
        configuration_file: Union[str, dict],
        preprocess_file: Optional[str] = None,
        requirements_file: Optional[str] = None,
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
            Path of the configuration file, but it could be a dict
        preprocess_file: Optional[str], optional
            Path of the preprocess script
        requirements_file: str
            Path of the requirements file

        Raises
        ------
        InputError
            Invalid parameters for model creation

        Returns
        -------
        str
            Model id (hash)

        Example
        -------
        >>> model.register_monitoring('parse', 'get_shap', configuration_file=PATH+'configuration.json', preprocess_file=PATH+'preprocess.py', requirements_file=PATH+'requirements.txt')
        """
        url = f"{self.base_url}/monitoring/register/{self.group}/{self.model_id}"

        if isinstance(configuration_file, str):
            with open(configuration_file, "rb") as f:
                conf_dict = json.load(f)

            conf = open(configuration_file, "rb")

        elif isinstance(configuration_file, dict):
            conf = json.dumps(configuration_file)
            conf_dict = configuration_file

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

        response = requests.post(
            url,
            data=form_data,
            files=upload_data,
            headers={
                "Authorization": "Bearer "
                + refresh_token(*self.credentials, self.base_url),
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.register_monitoring.__qualname__,
            },
        )

        if response.status_code == 201:
            data = response.json()
            model_id = data["ModelHash"]
            period = conf_dict["Period"]
            logger.info(f'{data["Message"]} - Hash: "{model_id}"')

            self.__host_monitoring(group=self.group, model_id=model_id, period=period)
            self.__host_monitoring_status(
                group=self.group, model_id=model_id, period=period
            )

            return model_id

        formatted_msg = parse_json_to_yaml(response.json())

        if response.status_code == 401:
            logger.error(
                "Login or password are invalid, please check your credentials."
            )
            raise AuthenticationError("Login not authorized.")

        if response.status_code >= 500:
            logger.error("Server is not available. Please, try it later.")
            raise ServerError("Server is not available!")

        logger.error(f"Upload error:\n{formatted_msg}")
        raise InputError("Invalid parameters for model creation")

    def model_info(self) -> None:
        """Show the model data in a better format"""
        logger.info(f"Result:\n{parse_json_to_yaml(self.model_data)}")


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

            return model.model_id

        client = MLOpsModelClient('123456')
        client.create_group('ex_group', 'Group for example purpose')

        data_path = './samples/syncModel/'

        model_id = new_sync_model(client, 'ex_group', data_path)

        model_list = client.search_models()
        print(model_list)

        model = client.get_model(model_id, 'ex_group')

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

            return model.model_id

        def run_model(client, model_id, data_path):
            model = client.get_model(model_id, 'ex_group')

            execution = model.predict(data_path+'input.csv')

            return execution

        client = MLOpsModelClient('123456')
        client.create_group('ex_group', 'Group for example purpose')

        data_path = './samples/asyncModel/'

        model_id = new_async_model(client, 'ex_group', data_path)

        execution = run_model(client, model_id, data_path)

        execution.get_status()

        execution.download_result()
    """

    def __repr__(self) -> str:
        return f'API version {self.version} - MLOpsModelClient(url="{self.base_url}", Token="{self.user_token}")'

    def __str__(self):
        return f"MLOPS {self.base_url} Model client:{self.user_token}"

    def __get_model_status(self, model_id: str, group: str) -> dict:
        """
        Gets the status of the model with the hash equal to `model_id`

        Parameters
        ----------
        group: str
            Group the model is inserted
        model_id: str
            Model id (hash) from the model being searched

        Raises
        ------
        ModelError
            Model unavailable

        Returns
        -------
        dict
            The model status and a message if the status is 'Failed'
        """

        url = f"{self.base_url}/model/status/{group}/{model_id}"
        response = requests.get(
            url,
            headers={
                "Authorization": "Bearer "
                + refresh_token(*self.credentials, self.base_url)
            },
        )
        if response.status_code not in [200, 410]:
            raise ModelError(f'Model "{model_id}" not found')

        return response.json()

    def get_model(
        self,
        *,
        model_id: str,
        group: str,
        group_token: Optional[str] = None,
        wait_for_ready: Optional[bool] = True,
    ) -> MLOpsModel:
        """
        Acess a model using its id

        Parameters
        ----------
        model_id: str
            Model id (hash) that needs to be acessed
        group: str
            Group the model is inserted.
        group_token: Optional[str], optional
            Token for executing the model (show when creating a group). It can be informed when getting the model or when running predictions, or using the env variable MLOPS_GROUP_TOKEN
        wait_for_ready: Optional[bool], optional
            If the model is being deployed, wait for it to be ready instead of failing the request. Defaults to True

        Raises
        ------
        ModelError
            Model unavailable
        ServerError
            Unknown return from server

        Returns
        -------
        MLOpsModel
            A MLOpsModel instance with the model hash from `model_id`

        Example
        -------
        >>> model.get_model(model_id='M9c3af308c754ee7b96b2f4a273984414d40a33be90242908f9fc4aa28ba8ec4', group='ex_group')
        """
        try:
            response = self.__get_model_status(model_id, group)
        except KeyError:
            raise ModelError("Model not found")

        status = response["Status"]

        if status == "Building":
            if wait_for_ready:
                print("Waiting for deploy to be ready.", end="")
                while status == "Building":
                    response = self.__get_model_status(model_id, group)
                    status = response["Status"]
                    print(".", end="", flush=True)
                    sleep(10)
            else:
                logger.info("Returning model, but model is not ready.")
                MLOpsModel(
                    model_id=model_id,
                    login=self.credentials[0],
                    password=self.credentials[1],
                    group=group,
                    url=self.base_url,
                    group_token=group_token,
                )

        if status in ["Disabled", "Ready"]:
            raise ModelError(
                f'Model "{model_id}" unavailable (disabled or deploy process is incomplete)'
            )
        elif status == "Failed":
            logger.error(str(response["Message"]))
            raise ModelError(
                f'Model "{model_id}" deploy failed, so model is unavailable.'
            )
        elif status == "Deployed":
            logger.info(f"Model {model_id} its deployed. Fetching model.")
            return MLOpsModel(
                model_id=model_id,
                login=self.credentials[0],
                password=self.credentials[1],
                group=group,
                url=self.base_url,
                group_token=group_token,
            )
        else:
            raise ServerError("Unknown model status: ", status)

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
        name: Optional[str], optional
            Text that it's expected to be on the model name. It runs similar to a LIKE query on SQL
        state: Optional[str], optional
            Text that it's expected to be on the state. It runs similar to a LIKE query on SQL
        group: Optional[str], optional
            Text that it's expected to be on the group name. It runs similar to a LIKE query on SQL
        only_deployed: Optional[bool], optional
            If it's True, filter only models ready to be used (status == "Deployed"). Defaults to False

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

        response = requests.get(
            url,
            params=query,
            headers={
                "Authorization": "Bearer "
                + refresh_token(*self.credentials, self.base_url),
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.search_models.__qualname__,
            },
        )

        if response.status_code == 200:
            results = response.json()["Results"]
            parsed_results = []
            for r in results:
                if schema := r.get("Schema"):
                    r["Schema"] = json.loads(schema)
                parsed_results.append(r)

            return [
                MLOpsModel(
                    model_id=m["ModelHash"],
                    login=self.credentials[0],
                    password=self.credentials[1],
                    group=m["Group"],
                    url=self.base_url,
                )
                for m in parsed_results
            ]

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
        raise ModelError("Could not search the model")

    def get_logs(
        self,
        *,
        model_id,
        start: Optional[str] = None,
        end: Optional[str] = None,
        routine: Optional[str] = None,
        type: Optional[str] = None,
    ):
        """
        Get the logs

        Parameters
        ----------
        model_id: str
            Model id (hash)
        start: str, optional
            Date to start filter. At the format aaaa-mm-dd
        end: str, optional
            Date to end filter. At the format aaaa-mm-dd
        routine: str, optional
            Type of routine being executed, can assume values 'Host' (for deployment logs) or 'Run' (for execution logs)
        type: str, optional
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
        >>> model.get_logs(routine='Run')
         {'Results':
            [{'ModelHash': 'B4c3af308c3e452e7b96b2f4a273984414d40a33be90242908f9fc4aa28ba8ec4',
                'RegisteredAt': '2023-02-03T16:06:45.5955220Z',
                'OutputType': 'Ok',
                'OutputData': '',
                'Routine': 'Run'}]
         }
        """
        url = f"{self.base_url}/model/logs/{model_id}"
        return self._logs(
            url=url,
            credentials=self.credentials,
            start=start,
            end=end,
            routine=routine,
            type=type,
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
            Path of the requirements file. The packages versions must be fixed eg: pandas==1.0
        schema: Union[str, dict], optional
            Path to a JSON or XML file with a sample of the input for the entrypoint function. A dict with the sample input can be sending as well
        group: Optional[str], optional
            Group the model is inserted.
        extra_files: Optional[list], optional
            A optional list with additional files paths that should be uploaded. If the scoring function refer to this file they will be on the same folder as the source file
        env: str, optional
            Flag that choose which environment (dev, staging, production) of MLOps you are using. Default is True
        python_version: Optional[str], optional
            Python version for the model environment. Available versions are 3.8, 3.9, 3.10. Defaults to '3.10'
        operation: Optional[str], optional
            Defines which kind operation is being executed (Sync or Async). Default value is Sync
        input_type: str
            The type of the input file that should be 'json', 'csv', 'parquet', 'txt', 'xls', 'xlsx'

        Raises
        ------
        InputError
            Some input parameters is invalid

        Returns
        -------
        str
            The new model id (hash)
        """

        url = f"{self.base_url}/model/upload/{group}"

        file_extesions = {"py": "script.py", "ipynb": "notebook.ipynb"}

        upload_data = [
            (
                "source",
                (file_extesions[source_file.split(".")[-1]], open(source_file, "rb")),
            ),
            ("model", (model_file.split("/")[-1], open(model_file, "rb"))),
            ("requirements", ("requirements.txt", open(requirements_file, "rb"))),
        ]

        if schema:
            upload_data.append(("schema", (schema, parse_dict_or_file(schema))))
        else:
            raise InputError("Schema file is mandatory")

        if operation == "Sync":
            input_type = "json"
        else:
            if input_type == "json|csv|parquet":
                raise InputError("Choose a input type from " + input_type)

        if env:
            upload_data.append(("env", (".env", open(env, "r"))))

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
            "python_version": "Python" + python_version.replace(".", ""),
        }

        response = requests.post(
            url,
            data=form_data,
            files=upload_data,
            headers={
                "Authorization": "Bearer "
                + refresh_token(*self.credentials, self.base_url),
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.__upload_model.__qualname__,
            },
        )

        if response.status_code == 201:
            data = response.json()
            model_id = data["ModelHash"]
            logger.info(f'{data["Message"]} - Hash: "{model_id}"')
            return model_id

        formatted_msg = parse_json_to_yaml(response.json())

        if response.status_code == 401:
            logger.error(
                "Login or password are invalid, please check your credentials."
            )
            raise AuthenticationError("Login not authorized.")

        if response.status_code >= 500:
            logger.error("Server is not available. Please, try it later.")
            raise ServerError("Server is not available!")

        logger.error(f"Upload error:\n{formatted_msg}")
        raise InputError("Invalid parameters for model creation")

    def __host_model(self, *, operation: str, model_id: str, group: str) -> None:
        """
        Builds the model execution environment

        Parameters
        ----------
        operation: str
            The model operation type (Sync or Async)
        model_id: str
            The uploaded model id (hash)
        group: str
            Group the model is inserted. Default is 'datarisk' (public group)

        Raises
        ------
        InputError
            Some input parameters is invalid
        """

        url = f"{self.base_url}/model/{operation}/host/{group}/{model_id}"

        response = requests.get(
            url,
            headers={
                "Authorization": "Bearer "
                + refresh_token(*self.credentials, self.base_url),
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.create_model.__qualname__,
            },
        )
        if response.status_code == 202:
            logger.info(f"Model host in process - Hash: {model_id}")
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
        raise InputError("Invalid parameters for model creation")

    def create_model(
        self,
        *,
        model_name: str,
        model_reference: str,
        source_file: str,
        model_file: str,
        requirements_file: str,
        group: str,
        schema: Optional[Union[str, dict]] = None,
        extra_files: Optional[list] = None,
        env: Optional[str] = None,
        python_version: str = "3.10",
        operation="Sync",
        input_type: str = "json|csv|parquet",
        wait_for_ready: bool = True,
    ) -> Union[MLOpsModel, str]:
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
        python_version: str, optional
            Python version for the model environment. Available versions are 3.8, 3.9, 3.10. Defaults to '3.10'
        operation: str
            Defines which kind operation is being executed (Sync or Async). Default value is Sync
        input_type: str
            The type of the input file that should be 'json', 'csv' or 'parquet'
        wait_for_ready: bool, optional
            Wait for model to be ready and returns a MLOpsModel instance with the new model. Defaults to True

        Raises
        ------
        InputError
            Some input parameters is invalid

        Returns
        -------
        Union[MLOpsModel, str]
            Returns the new model, if wait_for_ready=True runs the deployment process synchronously. If it's False, returns nothing after sending all the data to server and runs the deployment asynchronously

        Example
        -------
        >>> model = client.create_model('Model Example Sync', 'score',  './samples/syncModel/app.py', './samples/syncModel/'model.pkl', './samples/syncModel/requirements.txt','./samples/syncModel/schema.json', group=group, operation="Sync")
        """

        validate_python_version(python_version)
        validate_group_existence(group, self)

        model_id = self.__upload_model(
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

        self.__host_model(operation=operation.lower(), model_id=model_id, group=group)

        return self.get_model(
            model_id=model_id, group=group, wait_for_ready=wait_for_ready
        )

    def get_model_execution(
        self, *, model_id: str, exec_id: str, group: Optional[str] = None
    ) -> MLOpsExecution:
        """
        Get an execution instace (Async model only).

        Parameters
        ----------
        model_id: str
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
        >>> model.get_model_execution( model_id='M9c3af308c754ee7b96b2f4a273984414d40a33be90242908f9fc4aa28ba8ec4', exec_id = '1')
        """
        return self.get_model(model_id=model_id, group=group).get_model_execution(
            exec_id
        )
