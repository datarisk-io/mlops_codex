import json
from typing import Dict, Union

import requests

from mlops_codex.__utils import parse_json_to_yaml
from mlops_codex.base import BaseMLOps, BaseMLOpsClient
from mlops_codex.dataset import MLOpsDataset, MLOpsDatasetClient
from mlops_codex.exceptions import (
    AuthenticationError,
    CredentialError,
    DatasetNotFoundError,
    InputError,
    ServerError,
)
from mlops_codex.http_request_handler import make_request, refresh_token
from mlops_codex.logger_config import get_logger

logger = get_logger()


class MLOpsDataSourceClient(BaseMLOpsClient):
    """
    Class for client for manage datasources

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

    def register_datasource(
        self,
        *,
        datasource_name: str,
        provider: str,
        cloud_credentials: Union[dict, str],
        group: str,
    ):
        """
        Register the user cloud credentials to allow MLOps to use the provider to download the datasource.

        Parameters
        ----------
        group: str
            Name of the group where we will search the datasources.
        datasource_name: str
            Name given previously to the datasource.
        provider: str
            It can be "Azure", "AWS" or "GCP"
        cloud_credentials: str | Union[dict,str]
            Path or dict to a JSON with the credentials to access the provider.

        Returns
        ----------
        MLOpsDataSource
            A MLOpsDataSource object

        Example
        -------
        >>> client.register_datasource(
        >>>     datasource_name='MyDataSourceName',
        >>>     provider='GCP',
        >>>     cloud_credentials='./gcp_credentials.json',
        >>>     group='my_group'
        >>> )
        """

        datasource = MLOpsDataSource(
            datasource_name=datasource_name,
            provider=provider,
            group=group,
            login=self.credentials[0],
            password=self.credentials[1],
            url=self.base_url,
        )

        url = f"{self.base_url}/datasource/register/{group}"

        if isinstance(cloud_credentials, dict):
            credential_path = self.credentials_to_json(cloud_credentials)

            with open(credential_path, encoding="utf-8", mode="w") as credential_file:
                json.dump(datasource.credentials, credential_file)
        else:
            credential_path = cloud_credentials

        form_data = {"name": datasource_name, "provider": provider}

        files = {
            "credentials": (
                cloud_credentials.split("/")[-1],
                open(credential_path, "rb"),
            )
        }
        token = refresh_token(*self.credentials, self.base_url)

        response = requests.post(
            url=url,
            data=form_data,
            files=files,
            headers={
                "Authorization": "Bearer " + token,
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.register_datasource.__qualname__,
            },
            timeout=60,
        )

        if response.status_code == 200:
            logger.info(response.json().get("Message"))
            return datasource
        elif response.status_code == 400:
            del datasource
            if "Database produced an unexpected error" in response.text:
                raise ServerError("Database produced an unexpected error")
            if "not in the master group" in response.text:
                raise AuthenticationError("User is not in the master group.")
        raise CredentialError("Cloud Credential Error")

    # TODO: turn into staticmethod or external function
    def credentials_to_json(self, input_data: dict) -> str:
        """
        Transform dict to json.

        Parameters
        ----------
        input_data: dict
            A dictionary to save.

        Returns
        -------
        str
            Path to the credentials file.
        """

        path = "./credentials.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(input_data, f)
        return path

    def list_datasources(self, *, provider: str, group: str):
        """
        List all datasources of the group with this provider type.

        Parameters
        ----------
        group: str
            Name of the group where we will search the datasources
        provider: str ("Azure" | "AWS" | "GCP")

        Raises
        ------
        AuthenticationError
            Raised if there is an authentication issue.
        ServerError
            Raised if the server encounters an issue.
        InputError
            Raised if something went wrong.

        Returns
        ----------
        list
            A list of datasources information.

        Example
        -------
        >>> client.list_datasources(provider='GCP', group='my_group')
        """
        url = f"{self.base_url}/datasource/list?group={group}&provider={provider}"

        token = refresh_token(*self.credentials, self.base_url)

        response = requests.get(
            url=url,
            headers={
                "Authorization": "Bearer " + token,
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.list_datasources.__qualname__,
            },
            timeout=60,
        )
        if response.status_code == 200:
            results = response.json().get("Results")
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
        raise InputError("Bad Input. Client error")

    def get_datasource(self, *, datasource_name: str, provider: str, group: str):
        """
        Get a MLOpsDataSource to make datasource operations.

        Parameters
        ----------
        datasource_name: str
            Name given previously to the datasource.
        provider: str
            It can be "Azure", "AWS" or "GCP"
        group: str
            Name of the group where we will search the datasources

        Returns
        ----------
        MLOpsDataSource
            A MLOpsDataSource object

        Example
        -------
        >>> client.get_datasource(datasource_name='MyDataSourceName', provider='GCP', group='my_group')
        """
        datasources = self.list_datasources(provider=provider, group=group)
        for datasource in datasources:
            if datasource_name == datasource.get("Name"):
                return MLOpsDataSource(
                    datasource_name=datasource.get("Name"),
                    provider=datasource.get("Provider"),
                    group=datasource.get("Group"),
                    login=self.credentials[0],
                    password=self.credentials[1],
                    url=self.base_url,
                )
        raise InputError("Datasource not found!")


class MLOpsDataSource(BaseMLOps):
    """
    Class to operate actions in a datasource.

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
    datasource_name: str
        Name given previously to the datasource.
    provider: str
        Providers name, currently, MLOps supports:
        Azure Blob Storage as "Azure",
        AWS S3 as "AWS",
        Google GCP as "GCP".
    group: str
        Name of the group where we will search the datasources
    """

    def __init__(
        self,
        *,
        datasource_name: str,
        provider: str,
        group: str,
        login: str,
        password: str,
        url: str,
    ) -> None:
        super().__init__(login=login, password=password, url=url)
        self.datasource_name = datasource_name
        self.provider = provider
        self.group = group
        self.__datasets = MLOpsDatasetClient(login=login, password=password, url=url)

    def import_dataset(
        self, *, dataset_uri: str, dataset_name: str, force: bool = False
    ) -> Union[MLOpsDataset, Dict]:
        """
        Import a dataset inside a datasource.

        Parameters
        ----------
        dataset_uri: str
            Datasource cloud URI path.
        dataset_name: str
            The dataset defined name
        force: bool
            Optional[boolean]: when it is true it will force the datasource download from the provider.

        Returns
        ----------
        MLOpsDataset
            A MLOpsDataset with the identifier as dataset_hash.

        Raises
        ----------
        AuthenticationError
            Raised if there is an authentication issue.
        ServerError
            Raised if the server encounters an issue.
        InputError
            If any data sent is invalidated on server.

        Example
        -------
        >>> dataset = datasource.import_dataset(
        >>>     dataset_uri='https://storage.cloud.google.com/your-name/file.csv',
        >>>     dataset_name='meudataset'
        >>> )
        """
        form_data = {"uri": dataset_uri, "name": dataset_name}

        force = str(force).lower()

        token = refresh_token(*self.credentials, self.base_url)
        url = f"{self.base_url}/datasource/import/{self.group}/{self.datasource_name}?force={force}"
        response = requests.post(
            url=url,
            data=form_data,
            headers={
                "Authorization": "Bearer " + token,
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.import_dataset.__qualname__,
            },
            timeout=60,
        )

        if response.status_code == 200:
            datasets = response.json().get("Datasets")
            if len(datasets) == 1:
                dataset_hash = datasets[0]
                dataset = MLOpsDataset(
                    login=self.credentials[0],
                    password=self.credentials[1],
                    base_url=self.base_url,
                    hash=dataset_hash,
                    dataset_name=dataset_name,
                    group=self.group,
                )
                return dataset
            else:
                dts = {}
                for i, ds in enumerate(datasets):
                    dataset = MLOpsDataset(
                        login=self.credentials[0],
                        password=self.credentials[1],
                        base_url=self.base_url,
                        hash=ds,
                        dataset_name=dataset_name + f"_{i}",
                        group=self.group,
                    )
                    dts[f"dataset_{i}"] = dataset
                return dts

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
        raise InputError("Bad Input. Client error")

    def delete(self):
        """
        Delete the datasource on mlops. Pay attention when doing this action, it is irreversible!

        Example
        -------
        >>> datasource.delete()
        """
        url = f"{self.base_url}/datasources/{self.group}/{self.datasource_name}"

        token = refresh_token(*self.credentials, self.base_url)
        response = requests.delete(
            url=url,
            headers={
                "Authorization": "Bearer " + token,
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.delete.__qualname__,
            },
            timeout=60,
        )
        logger.info(response.json().get("Message"))

    def get_dataset(self, *, dataset_hash: str):
        """
        Get a MLOpsDataset to make dataset operations.

        Parameters
        ----------
        dataset_hash: str
            Name given previously to the datasource.

        Returns
        ----------
        MLOpsDataset
            A MLOpsDataset with the identifier as dataset_hash.

        Raises
        ----------
        DatasetNotFoundError
            When the dataset was not found

        Example
        ----------
        >>> dataset = datasource.get_dataset(dataset_hash='D589654eb26c4377b0df646e7a5675fa3c7d49575e03400b940dd5363006fc3a')
        """

        dataset_list = self.__datasets.list_datasets(
            origin="Datasource", datasource_name=self.datasource_name
        )

        for dataset in dataset_list:
            if dataset_hash == dataset.get("Hash"):
                return MLOpsDataset(
                    login=self.credentials[0],
                    password=self.credentials[1],
                    base_url=self.base_url,
                    hash=dataset.get("Hash"),
                    dataset_name=dataset.get("Name"),
                    group=self.group,
                )
        raise DatasetNotFoundError("Dataset hash not found!")

    def list_datasets(self) -> None:
        """
        Show datasets with Datasource origin
        """
        dataset_list = self.__datasets.list_datasets(
            origin="Datasource", datasource_name=self.datasource_name
        )
        for dataset in dataset_list:
            print(parse_json_to_yaml(dataset))

    def get_status(self, group: str, dataset_hash: str) -> dict:
        """
        Get dataset status.

        Parameters
        ----------
        group: str
            Name of the group where we will search the datasources
        dataset_hash: str
            Name given previously to the datasource.

        Returns
        ----------
        dict
        Dictionary with the status and log of the dataset.

        Raises
        ----------
        DatasetNotFound
            When the dataset was not found

        Example
        ----------
        >>> dataset.get_status()
        """
        url = f"{self.base_url}/datasets/status/{group}/{dataset_hash}"
        token = refresh_token(*self.credentials, self.base_url)
        response = make_request(
            url=url,
            method="GET",
            success_code=200,
            custom_exception=DatasetNotFoundError,
            custom_exception_message=f"Dataset not found for hash {dataset_hash}.",
            specific_error_code=404,
            headers={
                "Authorization": "Bearer " + token,
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.get_status.__qualname__,
            },
        )

        status = response.json().get("Status")
        log = response.json().get("Log")

        return {"status": status, "log": log}
