"""
External Monitoring Module
"""

from datetime import datetime
from time import sleep
from typing import NamedTuple, Optional

import requests

from mlops_codex.__model_states import MonitoringStatus
from mlops_codex.__utils import parse_json_to_yaml, validate_kwargs
from mlops_codex.base import BaseMLOps, BaseMLOpsClient
from mlops_codex.exceptions import (
    AuthenticationError,
    ExecutionError,
    ExternalMonitoringError,
    GroupError,
    InputError,
    ServerError,
)
from mlops_codex.http_request_handler import refresh_token
from mlops_codex.logger_config import get_logger
from mlops_codex.validations import validate_python_version

logger = get_logger()


class MLOpsExternalMonitoring(BaseMLOps):
    """
    Class that handles an external monitoring object
    """

    def __init__(
        self,
        group: str,
        ex_monitoring_hash: str,
        status: Optional[MonitoringStatus] = MonitoringStatus.Unvalidated,
        login: Optional[str] = None,
        password: Optional[str] = None,
        url: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        group:
        ex_monitoring_hash:
        status:
        login:
        password:
        url:
        """

        super().__init__(login=login, password=password, url=url)
        self.external_monitoring_url = f"{self.base_url}/external-monitoring"
        self.ex_monitoring_hash = ex_monitoring_hash
        self.group = group
        self.status = status

    def __repr__(self):
        return f"Group: {self.group}\nHash: {self.ex_monitoring_hash}\nStatus: {self.status}"

    def __str__(self):
        return f"Group: {self.group}\nHash: {self.ex_monitoring_hash}\nStatus: {self.status}"

    def _upload_file(
        self,
        field: str,
        file: str,
        url: str,
        form: Optional[dict] = None,
    ) -> bool:
        """Upload a file

        Args:
            field (str): Field name
            file (str): File to upload
            url (str): Url to register the external monitoring
            form (Optional[dict]): Dict with form data

        Raises:
            AuthenticationError
            GroupError
            ServerError
            ExternalMonitoringError

        Returns:
            bool: True if file was successfully uploaded
        """
        file_extensions = {"py": "script.py", "ipynb": "notebook.ipynb"}

        file_name = file.split("/")[-1]

        if file.endswith(".py") or file.endswith(".ipynb"):
            file_name = file_extensions[file.split(".")[-1]]

        upload_data = [(field, (file_name, open(file, "rb")))]
        response = requests.patch(
            url,
            data=form,
            files=upload_data,
            headers={
                "Authorization": "Bearer "
                + refresh_token(*self.credentials, self.base_url),
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.upload_file.__qualname__,
            },
            timeout=60,
        )

        formatted_msg = parse_json_to_yaml(response.json())

        if response.status_code == 201:
            logger.debug(f"File uploaded successfully:\n{formatted_msg}")
            return True

        if response.status_code == 401:
            logger.debug(
                "Login or password are invalid, please check your credentials."
            )
            raise AuthenticationError("Login not authorized.")

        if response.status_code == 404:
            logger.debug("Group not found in the database")
            raise GroupError("Group not found in the database")

        if response.status_code >= 500:
            logger.debug("Server is not available. Please, try it later.")
            raise ServerError("Server is not available!")

        logger.debug(f"Something went wrong...\n{formatted_msg}")
        raise ExternalMonitoringError("Could not register the monitoring.")

    def upload_file(
        self,
        *,
        model_file: Optional[str] = None,
        requirements_file: Optional[str] = None,
        preprocess_file: Optional[str] = None,
        preprocess_reference: Optional[str] = None,
        shap_reference: Optional[str] = None,
        python_version: Optional[str] = "3.10",
    ):
        """
        Validate inputs before sending files.

        Parameters
        ----------
        model_file: Optional[str], optional
            Path to your `model.pkl` file. Defaults to None.
        requirements_file: Optional[str], optional
            Path to your `requirements.txt` file. Defaults to None.
        preprocess_file: Optional[str], optional
            Path to your preprocessing file. Defaults to None.
        preprocess_reference: Optional[str], optional
            Preprocessing function entrypoint. Defaults to None.
        shap_reference: Optional[str], optional
            Shap function entrypoint. Defaults to None.
        python_version: Optional[str], optional
            Python version. Can be "3.8", "3.9", or "3.10". Defaults to "3.10".

        Raises
        ------
        InputError
            Raised if there is an error with the input `model_file`.
        InputError
            Raised if there is an error with the input `requirements_file`.
        """

        if model_file is not None:
            missing_args = [
                f
                for f in [
                    model_file,
                    requirements_file,
                    preprocess_file,
                    preprocess_reference,
                    shap_reference,
                    python_version,
                ]
                if f is None
            ]
            if missing_args:
                logger.error(f"You must pass the following arguments: {missing_args}")
                raise InputError("Missing files, function entrypoint or python version")

        if preprocess_file is not None:
            missing_args = [
                f
                for f in [
                    requirements_file,
                    preprocess_file,
                    preprocess_reference,
                    shap_reference,
                    python_version,
                ]
                if f is None
            ]
            if missing_args:
                logger.error(f"You must pass the following arguments: {missing_args}")
                raise InputError("Missing files, function entrypoint or python version")

        python_version= validate_python_version(python_version)

        uploads = [
            ("model", model_file, "model-file", None),
            ("requirements", requirements_file, "requirements-file", None),
            (
                "script",
                preprocess_file,
                "script-file",
                {
                    "preprocess_reference": preprocess_reference,
                    "shap_reference": shap_reference,
                    "python_version": python_version,
                },
            ),
        ]

        for field, file, path, form in uploads:
            if file is not None:
                url = f"{self.external_monitoring_url}/{self.ex_monitoring_hash}/{path}"
                self._upload_file(field, file, url, form)
                logger.info(f"{file} file uploaded successfully")

    def host(self, wait: Optional[bool] = False):
        """
        Host the new external monitoring.

        Parameters
        ----------
        wait: Optional[bool], optional
            If true, wait until the host is validated or invalidate.

        Raises
        ------
        AuthenticationError
            Raised if there is an authentication issue.
        GroupError
            Raised if there is an error related to the group.
        ServerError
            Raised if the server encounters an issue.
        ExternalMonitoringError
            Raised if there is an error specific to external monitoring.

        Returns
        -------
        str
            The external monitoring hash if the new external monitoring is successfully hosted.
        """

        if self.status == MonitoringStatus.Validated:
            logger.info(
                f"You can't host a model that is already hosted. Status is {self.status}"
            )
            return

        response = requests.patch(
            url=f"{self.external_monitoring_url}/{self.ex_monitoring_hash}/status",
            headers={
                "Authorization": "Bearer "
                + refresh_token(*self.credentials, self.base_url),
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.host.__qualname__,
            },
            timeout=60,
        )

        formatted_msg = parse_json_to_yaml(response.json())
        if response.status_code == 202:
            self.status = MonitoringStatus.Validating
            if wait:
                self.wait_ready()
            logger.info("Hosted external monitoring successfully")
            return self.ex_monitoring_hash

        if response.status_code == 401:
            logger.error(
                "Login or password are invalid, please check your credentials."
            )
            raise AuthenticationError("Login not authorized.")

        if response.status_code == 404:
            logger.error("Group not found in the database")
            raise GroupError("Group not found in the database")

        if response.status_code >= 500:
            logger.error(f"Something went wrong...\n{formatted_msg}")
            raise ServerError("Server Error. Could not register the monitoring.")

        raise ExternalMonitoringError("Unknown error. Please contact administrator.")

    def wait_ready(self):
        """
        Check the status of the external monitoring.

        Returns
        -------
        str
            The status of the external monitoring.
        """
        response = requests.get(
            url=f"{self.external_monitoring_url}/{self.ex_monitoring_hash}/status",
            headers={
                "Authorization": "Bearer "
                + refresh_token(*self.credentials, self.base_url),
            },
            timeout=60,
        )
        message = response.json()
        status = message["Status"]

        print("Waiting the monitoring host...", end="")

        while status not in [MonitoringStatus.Validated, MonitoringStatus.Invalidated]:
            response = requests.get(
                url=f"{self.external_monitoring_url}/{self.ex_monitoring_hash}/status",
                headers={
                    "Authorization": "Bearer "
                    + refresh_token(*self.credentials, self.base_url),
                },
                timeout=60,
            )
            message = response.json()
            status = message["Status"]

            formatted_msg = parse_json_to_yaml(response.json())
            if response.status_code == 401:
                logger.debug(
                    "Login or password are invalid, please check your credentials."
                )
                raise AuthenticationError("Login not authorized.")

            if response.status_code == 404:
                logger.debug("Group not found in the database")
                raise GroupError("Group not found in the database")

            if response.status_code >= 500:
                logger.debug("Server is not available. Please, try it later.")
                raise ServerError("Server is not available!")

            if response.status_code > 300:
                logger.debug(f"Something went wrong...\n{formatted_msg}")
                raise ExternalMonitoringError(
                    "Unexpected error. Could not register the monitoring."
                )

            print(".", end="", flush=True)
            sleep(30)

        if status == MonitoringStatus.Invalidated:
            res_message = message["Message"]
            self.status = MonitoringStatus.Invalidated
            logger.debug(f"Model monitoring host message: {res_message}")
            raise ExecutionError("Monitoring host failed")

        self.status = MonitoringStatus.Validated
        logger.debug(
            f'External monitoring host validated - Hash: "{self.ex_monitoring_hash}"'
        )

    def logs(self, start: str, end: str):
        """
        Get the logs of an external monitoring.

        Parameters
        ----------
        start: str
            Start date to look for the records. The format must be `dd-MM-yyyy`.
        end: str
            End date to look for the records. The format must be `dd-MM-yyyy`.
        """
        url = f"{self.base_url}/monitoring/search/records/{self.group}/{self.ex_monitoring_hash}"
        print(
            parse_json_to_yaml(
                self._logs(url=url, credentials=self.credentials, start=start, end=end)
            )
        )


class MLOpsExternalMonitoringClient(BaseMLOpsClient):
    """
    Class that handles MLOps External Monitoring Client

    Parameters
    ----------
    login: str
        Login for authenticating with the client.
        You can also use the env variable MLOPS_USER to set this
    password: str
        Password for authenticating with the client.
        You can also use the env variable MLOPS_PASSWORD to set this
    url: str
        URL to MLOps Server. Default value is https://neomaril.datarisk.net/, use it to test your deployment first before changing to production. You can also use the env variable MLOPS_URL to set this

    Raises
    ------
    ServerError
        Database produced an unexpected error.
    AuthenticationError
        If user is not in the master group.
    CredentialError
        If the Cloud Credential is Invalid
    """

    # TODO: It would be more appropriate to move this to an internal creation method, as its current placement seems illogical.
    #       btw, it is my mistake!!
    class ExternalMonitoringData(NamedTuple):
        """External monitoring data"""

        name: str
        group: str
        training_execution_id: int
        period: str
        input_cols: list
        output_cols: list
        datasource_name: str
        extraction_type: str
        datasource_uri: str
        column_name: Optional[str]
        reference_date: Optional[str]
        python_version: Optional[str]

    @validate_kwargs(ExternalMonitoringData)
    def validate(self, **kwargs):
        """Method to validate data
        NOTE: This method is necessary if I want to use the try/except in line 460.
              Maybe in the future would be nice to migrate this to a generic interface
        """
        pass

    def __repr__(self) -> str:
        return f"API version {self.version} - MLOpsExternalMonitoringClient"

    def __str__(self):
        return f"MLOPS {self.base_url} External Monitoring client:{self.user_token}"

    def __register(self, configuration: dict, url: str) -> str:
        """Register a new external monitoring

        Args:
            configuration dict: Dict with configuration
            url (str): Url to register the external monitoring

        Raises:
            AuthenticationError
            GroupError
            ServerError
            ExternalMonitoringError

        Returns:
            External monitoring Hash
        """
        response = requests.post(
            url,
            json=configuration,
            headers={
                "Authorization": "Bearer "
                + refresh_token(*self.credentials, self.base_url),
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.register_monitoring.__qualname__,
            },
            timeout=60,
        )
        formatted_msg = parse_json_to_yaml(response.json())

        if response.status_code == 201:
            logger.debug(
                f"External monitoring was successfully registered:\n{formatted_msg}"
            )
            external_monitoring_hash = response.json()["ExternalMonitoringHash"]
            return external_monitoring_hash

        if response.status_code == 401:
            logger.debug(
                "Login or password are invalid, please check your credentials."
            )
            raise AuthenticationError("Login not authorized.")

        if response.status_code > 401 and response.status_code < 500:
            logger.error(formatted_msg)
            raise InputError("Invalid inputs")

        if response.status_code == 404:
            logger.debug("Group not found in the database")
            raise GroupError("Group not found in the database")

        if response.status_code >= 500:
            logger.debug("Server is not available. Please, try it later.")
            raise ServerError("Server is not available!")

        logger.debug(f"Something went wrong...\n{formatted_msg}")
        raise ExternalMonitoringError("Could not register the monitoring.")

    def register_monitoring(self, **kwargs) -> Optional[MLOpsExternalMonitoring]:
        """
        Register a MLOps External Monitoring.

        Parameters
        ----------
        name: str
            External Monitoring name.
        group: str
            External Monitoring group. The group is the same used for the external training and datasource.
        training_execution_id: int
            Valid MLOps training execution id.
        period: str
            The frequency the monitoring will run. It can be one of the following: "Day", "Week", "Quarter",
            "Month", "Year".
        input_cols: list
            List with input column names.
        output_cols: list
            List with output column names.
        datasource_name: str
            Valid MLOps datasource name.
        extraction_type: str
            Type of extraction. It can be one of the following: "Incremental", "Full".
        datasource_uri: str
            Valid datasource URI.
        column_name: Optional[str], optional
            Column name of the data column.
        reference_date: Optional[str], optional
            Reference extraction date.
        python_version: Optional[str], optional
            Python version used to run preprocessing scripts. It can be one of the following: "3.8", "3.9", "3.10". Defaults to "3.10".

        Raises
        ------
        InputError
            Raised if there is an error with the provided input.

        Returns
        -------
        MLOpsExternalMonitoring
            The newly registered MLOps external monitoring instance.
        """

        try:
            self.validate(**kwargs)
        except ValueError as e:
            print("Validation error:", e)
            return
        except TypeError as e:
            print("Type error:", e)
            return

        base_external_url = f"{self.base_url}/external-monitoring"

        if kwargs["period"] not in ["Day", "Week", "Quarter", "Month", "Year"]:
            logger.error(
                f"{kwargs['period']} is not available. Must be Day | Week | Quarter | Month | Year"
            )
            raise InputError("Period is not valid")

        if kwargs["extraction_type"] not in ["Full", "Incremental"]:
            logger.error(
                f"{kwargs['extraction_type']} is not available. Must be 'Full' or 'Incremental'"
            )
            raise InputError("Extraction Type is not valid")

        configuration_file = {
            "Name": kwargs["name"],
            "Group": kwargs["group"],
            "TrainingExecutionId": kwargs["training_execution_id"],
            "Period": kwargs["period"],
            "InputCols": kwargs["input_cols"],
            "OutputCols": kwargs["output_cols"],
            "DataSourceName": kwargs["datasource_name"],
            "ExtractionType": kwargs["extraction_type"],
            "DataSourceUri": kwargs["datasource_uri"],
        }

        if kwargs.get("column_name"):
            configuration_file["ColumnName"] = kwargs["column_name"]

        if kwargs.get("reference_date"):
            try:
                datetime.strptime(kwargs.get("reference_date"), "%Y-%m-%d")
                configuration_file["ReferenceDate"] = kwargs.get("reference_date")
            except ValueError as exc:
                logger.error("Reference date is in incorrect format. Use 'YYYY-MM-DD'")
                raise InputError("Date is not in the correct format") from exc

        if kwargs.get("python_version"):
            python_version = validate_python_version(kwargs.get("python_version"))
            configuration_file["PythonVersion"] = python_version

        external_monitoring_hash = self.__register(
            configuration=configuration_file, url=base_external_url
        )
        external_monitoring = MLOpsExternalMonitoring(
            login=self.credentials[0],
            password=self.credentials[1],
            url=self.base_url,
            group=kwargs["group"],
            ex_monitoring_hash=external_monitoring_hash,
            status=MonitoringStatus.Unvalidated,
        )

        logger.info(
            f"External Monitoring registered successfully. Hash - {external_monitoring_hash}"
        )

        return external_monitoring

    def __list_external_monitoring(self):
        url = f"{self.base_url}/external-monitoring"
        response = requests.get(
            url=url,
            headers={
                "Authorization": "Bearer "
                + refresh_token(*self.credentials, self.base_url),
            },
            timeout=60,
        )

        if response.status_code == 401:
            logger.error(
                "Login or password are invalid, please check your credentials."
            )
            raise AuthenticationError("Login not authorized.")

        if response.status_code == 404:
            logger.error("Group not found in the database")
            raise GroupError("Group not found in the database")

        if response.status_code >= 500:
            logger.error("Server is not available. Please, try it later.")
            raise ServerError("Server is not available!")

        if response.status_code != 200:
            formatted_msg = parse_json_to_yaml(response.json())
            logger.error(f"Something went wrong...\n{formatted_msg}")
            raise ExternalMonitoringError("Could not register the monitoring.")

        for external_monitoring in response.json()["Result"]:
            yield external_monitoring

    def list_hosted_external_monitorings(self) -> None:
        """List all hosted external monitoring"""
        for result in self.__list_external_monitoring():
            print(parse_json_to_yaml(result))

    def get_external_monitoring(
        self, external_monitoring_hash: str
    ) -> MLOpsExternalMonitoring:
        """
        Return an external monitoring.

        Parameters
        ----------
        external_monitoring_hash: str
            External Monitoring Hash.

        Raises
        ------
        ExternalMonitoringError
            Raised if there is an error fetching the external monitoring.

        Returns
        -------
        MLOpsExternalMonitoring
            The requested MLOps external monitoring instance.
        """

        for external_monitoring_dict in self.__list_external_monitoring():
            if external_monitoring_dict["Hash"] == external_monitoring_hash:
                logger.info("External monitoring found")

                group = external_monitoring_dict["Group"]
                external_monitoring = MLOpsExternalMonitoring(
                    login=self.credentials[0],
                    password=self.credentials[1],
                    url=self.base_url,
                    group=group,
                    ex_monitoring_hash=external_monitoring_hash,
                )
                external_monitoring.wait_ready()
                return external_monitoring
        raise ExternalMonitoringError(
            f"External monitoring not found for {external_monitoring_hash}"
        )
