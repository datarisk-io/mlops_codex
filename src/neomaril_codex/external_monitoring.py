from typing import Optional, Union

import requests
from time import sleep

from neomaril_codex.__utils import parse_json_to_yaml, refresh_token
from neomaril_codex.base import BaseNeomaril, BaseNeomarilClient
from neomaril_codex.exceptions import (
    AuthenticationError,
    ExecutionError,
    ExternalMonitoringError,
    ServerError,
    GroupError,
    InputError
)
from neomaril_codex.logger_config import get_logger

logger = get_logger()


class NeomarilExternalMonitoring(BaseNeomaril):
    def __init__(
        self,
        group: str,
        ex_monitoring_hash: str,
        login: Optional[str] = None,
        password: Optional[str] = None,
        url: Optional[str] = None,
    ):
        super().__init__(login=login, password=password, url=url)
        self.ex_monitoring_hash = ex_monitoring_hash
        self.group = group

    def __repr__(self):
        return f"Group: {self.group}\nHash: {self.ex_monitoring_hash}"

    def __str__(self):
        return f"Group: {self.group}\nHash: {self.ex_monitoring_hash}"


class NeomarilExternalMonitoringClient(BaseNeomarilClient):
    def __repr__(self) -> str:
        return f'API version {self.version} - NeomarilExternalMonitoringClient(url="{self.base_url}", Token="{self.user_token}")'

    def __str__(self):
        return f"NEOMARIL {self.base_url} External Monitoring client:{self.user_token}"

    def __register(self, configuration_file: Union[str, dict], url: str) -> str:
        """Register a new external monitoring

        Attributes:
        -----------
        configuration_file (Union[str, dict]): Dict with configuration
        url (str): Url to register the external monitoring

        Raises:
        -------
        AuthenticationError
        GroupError
        ServerError
        ExternalMonitoringError

        Returns:
        --------
        str: External monitoring Hash
        """
        response = requests.post(
            url,
            json=configuration_file,
            headers={
                "Authorization": "Bearer " + refresh_token(*self.credentials, self.base_url),
            },
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

        if response.status_code == 404:
            logger.debug("Group not found in the database")
            raise GroupError("Group not found in the database")

        if response.status_code >= 500:
            logger.debug("Server is not available. Please, try it later.")
            raise ServerError("Server is not available!")

        logger.debug(f"Something went wrong...\n{formatted_msg}")
        raise ExternalMonitoringError("Could not register the monitoring.")

    def __upload_file(
        self, field: str, file: str, url: str, form: Optional[dict] = None
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
            },
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

    def __host(self, url: str) -> bool:
        """Host the new external monitoring

        Attributes:
        -----------
        url (str): Url to host the external monitoring

        Raises:
        -------
        AuthenticationError
        GroupError
        ServerError
        ExternalMonitoringError

        Returns
        -------
        bool: True if host the new external monitoring
        """
        response = requests.patch(
            url,
            headers={
                "Authorization": "Bearer "
                + refresh_token(*self.credentials, self.base_url),
            },
        )

        formatted_msg = parse_json_to_yaml(response.json())
        if response.status_code == 200:
            logger.debug("Hosted external monitoring successfully")
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

    def __status(self, url: str, external_monitoring_hash: str):
        """Check the status of the external monitoring

        Args:
            url (str): Url to check the status of the external monitoring
            external_monitoring_hash (str): External monitoring Hash

        Returns:
            str: external monitoring
        """
        response = requests.get(
            url,
            headers={
                "Authorization": "Bearer "
                 + refresh_token(*self.credentials, self.base_url),
            },
        )
        message = response.json()
        status = message["Status"]

        print("Waiting the monitoring host...", end="")

        while status not in ["Validated", "Invalidated"]:
            response = requests.get(
                url,
                headers={
                    "Authorization": "Bearer "
                    + refresh_token(*self.credentials, self.base_url),
                },
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

            if response.status_code != 200:
                logger.debug(f"Something went wrong...\n{formatted_msg}")
                raise ExternalMonitoringError(
                    "Unexpected error. Could not register the monitoring."
                )

            print(".", end="", flush=True)
            sleep(30)

        if status == "Invalidated":
            res_message = message["Message"]
            logger.debug(f"Model monitoring host message: {res_message}")
            raise ExecutionError("Monitoring host failed")

        logger.debug(
            f'External monitoring host validated - Hash: "{external_monitoring_hash}"'
        )
        return external_monitoring_hash

    def register_monitoring(
        self,
        *,
        configuration_file: Union[str, dict],
        model_file: str,
        group: Optional[str] = "datarisk",
        requirements_file: Optional[str] = None,
        preprocess_file: Optional[str] = None,
        preprocess_reference: Optional[str] = None,
        shap_reference: Optional[str] = None,
        python_version: Optional[str] = "3.10",
    ) -> NeomarilExternalMonitoring:
        """
        Register and host a Neomaril External Monitoring

        Attributes
        ----------
        configuration_file : str or dict
            Path or dict that represents the configuration file
        model_file: str
            Path to your locally trained model
        group: str
            Group the model is inserted. Default is 'datarisk' (public group)
        requirements_file : str
            Path of the requirements file
        preprocess_file : str, optional
            Path of the preprocess script.
        preprocess_reference : str
            Name of the preprocess reference
        shap_reference : str
            Name of the preprocess function
        python_version : str, optional
            The version of the python environment. Can be 3.8, 3.9 or 3.10. Default is 3.10

        Returns
        -------
        NeomarilExternalMonitoring
            A Neomaril External Monitoring
        """

        base_external_url = f"{self.base_url}/external-monitoring/{group}"

        if python_version not in ["3.8", "3.9", "3.10"]:
            raise InputError(
                "Invalid python version. Available versions are 3.8, 3.9, 3.10"
            )

        if preprocess_file is not None and (preprocess_reference is None or shap_reference is None):
            raise InputError(
                "You must pass the preprocess entrypoint and shap reference!"
            )

        python_version = "Python" + python_version.replace(".", "")

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

        external_monitoring_hash = self.__register(
            configuration_file=configuration_file, url=base_external_url
        )
        logger.info(f"External Monitoring registered successfully. Hash - {external_monitoring_hash}")

        for field, file, path, form in uploads:
            if file is not None:
                url = f"{base_external_url}/{external_monitoring_hash}/{path}"
                self.__upload_file(field, file, url, form)
        logger.info(f"Files uploaded successfully")

        self.__host(url=f"{base_external_url}/{external_monitoring_hash}/status")
        self.__status(
            url=f"{base_external_url}/{external_monitoring_hash}/status",
            external_monitoring_hash=external_monitoring_hash,
        )

        logger.info(f'External monitoring host validated')
        return NeomarilExternalMonitoring(
            login=self.credentials[0],
            password=self.credentials[1],
            url=self.base_url,
            group=group,
            ex_monitoring_hash=external_monitoring_hash,
        )

    def list_hosted_external_monitorings(self) -> None:
        """List all hosted external monitoring"""

        url = f"{self.base_url}/external-monitoring"
        response = requests.get(
            url=url,
            headers={
                "Authorization": "Bearer " + refresh_token(*self.credentials, self.base_url),
            },
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

        results = response.json()["Result"]
        count = response.json()["Count"]
        logger.info(f"Found {count} monitorings")
        for result in results:
            print(parse_json_to_yaml(result))

    def get_external_monitoring(self, group: str, external_monitoring_hash: str) -> NeomarilExternalMonitoring:
        url = f"{self.base_url}/external-monitoring/{group}/{external_monitoring_hash}"
        response = requests.get(
            url=url,
            headers={
                "Authorization": "Bearer " + refresh_token(*self.credentials, self.base_url),
            },
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

        logger.info(f'External monitoring found')
        return NeomarilExternalMonitoring(
            login=self.credentials[0],
            password=self.credentials[1],
            url=self.base_url,
            group=group,
            ex_monitoring_hash=external_monitoring_hash,
        )
