from dataclasses import dataclass, field
from typing import Optional

from mlops_codex.__utils import parse_json_to_yaml
from mlops_codex.base import BaseMLOpsClient
from mlops_codex.exceptions import DatasetNotFoundError
from mlops_codex.http_request_handler import make_request, refresh_token
from mlops_codex.logger_config import get_logger

logger = get_logger()

def validate_dataset(dataset):
    """
    Check if a dataset is a valid hash or a valid MLOps Dataset
    Args:
        dataset: dataset provided by the user

    Returns:
        str: if the dataset is a valid hash or a valid MLOps Dataset
    """

    if not isinstance(dataset, (MLOpsDataset, str)):
        raise TypeError("Dataset must be a MLOpsDataset or str")

    if isinstance(dataset, MLOpsDataset):
        dataset_hash = dataset.hash
    else:
        dataset_hash = dataset

    if not dataset_hash.startswith("D"):
        raise ValueError("The provided dataset hash is not valid. Check your dataset hash")

    return dataset_hash


class MLOpsDatasetClient(BaseMLOpsClient):
    """
    Class to operate actions in a dataset.

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

    def __query_datasets(
        self,
        *,
        origin: Optional[str] = None,
        origin_id: Optional[int] = None,
        datasource_name: Optional[str] = None,
        group: Optional[str] = None,
    ):
        url = f"{self.base_url}/datasets/list"
        token = refresh_token(*self.credentials, self.base_url)

        query = {}

        if group:
            query["group"] = group

        if origin and origin != "Datasource":
            query["origin"] = origin
            if origin_id:
                query["origin_id"] = origin_id

        if origin == "Datasource":
            query["origin"] = origin
            if datasource_name:
                query["datasource"] = datasource_name

        response = make_request(
            url=url,
            method="GET",
            success_code=200,
            headers={
                "Authorization": "Bearer " + token,
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.list_datasets.__qualname__,
            },
            params=query,
        )
        return response

    def load_dataset(self, dataset_hash: str):
        result = self.__query_datasets().json()["Results"]
        for r in result:
            if r["DatasetHash"] == dataset_hash:
                return MLOpsDataset(
                    login=self.credentials[0],
                    password=self.credentials[1],
                    base_url=self.base_url,
                    hash=dataset_hash,
                    dataset_name=r["Name"],
                    group=r["Group"],
                )

        logger.info(f"Dataset {dataset_hash} not found")

    def delete(self, group: str, dataset_hash: str) -> None:
        """
        Delete the dataset on mlops. Pay attention when doing this action, it is irreversible!

        Parameters
        ---------
        group: str
            Group to delete.
        dataset_hash: str
            Dataset hash to delete.

        Example
        ----------
        >>> dataset.delete()
        """
        url = f"{self.url}/datasets/{group}/{dataset_hash}"
        token = refresh_token(*self.credentials, self.base_url)
        make_request(
            url=url,
            method="DELETE",
            success_code=200,
            custom_exception=DatasetNotFoundError,
            custom_exception_message="Dataset not found.",
            specific_error_code=404,
            headers={
                "Authorization": "Bearer " + token,
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.delete.__qualname__,
            },
        )

        logger.info(f"Dataset {dataset_hash} deleted.")

    def list_datasets(
        self,
        *,
        origin: Optional[str] = None,
        origin_id: Optional[int] = None,
        datasource_name: Optional[str] = None,
        group: Optional[str] = None,
    ) -> None:
        """
        List datasets from datasources.

        Parameters
        ----------
        origin: Optional[str]
            Origin of a dataset. It can be "Training", "Preprocessing", "Datasource" or "Model"
        origin_id: Optional[str]
            Integer that represents the id of a dataset, given an origin
        datasource_name: Optional[str]
            Name of the datasource
        group: Optional[str]
            Name of the group where we will search the dataset

        Example
        -------
        >>> dataset.list_datasets()
        """
        response = self.__query_datasets(origin=origin, origin_id=origin_id, datasource_name=datasource_name, group=group)
        formatted_response = parse_json_to_yaml(response.json())
        print(formatted_response)


@dataclass(frozen=True)
class MLOpsDataset:
    """
    Dataset class to represent mlops dataset.

    Parameters
    ----------
    login: str
        Login for authenticating with the client.
    password: str
        Password for authenticating with the client.
    base_url: str
        URL to MLOps Server. Default value is https://neomaril.datarisk.net, use it to test your deployment first before changing to production. You can also use the env variable MLOPS_URL to set this
    hash: str
        Dataset hash to download.
    dataset_name: str
        Name of the dataset.
    group: str
        Name of the group where we will search the dataset
    origin: str
        Origin of the dataset. It can be "Training", "Preprocessing", "Datasource" or "Model"
    """

    login: str = field(repr=False)
    password: str = field(repr=False)
    base_url: str = field(repr=False)
    hash: str
    dataset_name: str
    group: str

    def download(
        self,
        path: Optional[str] = "./",
        filename: Optional[str] = "dataset",
    ) -> None:
        """
        Download a dataset from mlops. The dataset will be a csv or parquet file.

        Parameters
        ----------
        path: str, optional
            Path to the downloaded dataset. Defaults to './'.
        filename: str, optional
            Name of the downloaded dataset. Defaults to 'dataset.zip'.

        Raises
        ------
        AuthenticationError
            Raised if there is an authentication issue.
        DatasetNotFoundError
            Raised if there is no dataset with the given name.
        ServerError
            Raised if the server encounters an issue.
        """

        if not path.endswith("/"):
            path = path + "/"

        url = f"{self.base_url}/datasets/result/{self.group}/{self.hash}"
        token = refresh_token(self.login, self.password, self.base_url)
        response = make_request(
            url=url,
            method="GET",
            success_code=200,
            headers={
                "Authorization": "Bearer " + token,
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.download.__qualname__,
            },
        )

        try:
            response.content.decode("utf-8")
            filename += ".csv"
        except UnicodeDecodeError:
            filename += ".parquet"

        with open(path + filename, "wb") as dataset_file:
            dataset_file.write(response.content)

        logger.info(f"MLOpsDataset downloaded to {path + filename}")
