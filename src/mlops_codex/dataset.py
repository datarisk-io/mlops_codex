from typing import Any, List, Optional

from pydantic import BaseModel, Field, PrivateAttr

from mlops_codex.base import BaseMLOpsClient
from mlops_codex.exceptions import DatasetNotFoundError
from mlops_codex.http_request_handler import make_request, refresh_token
from mlops_codex.logger_config import get_logger

logger = get_logger()


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

    def __init__(self, login: str, password: str, url: str) -> None:
        super().__init__(login=login, password=password, url=url)

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
            custom_exception_message=f"Dataset not found.",
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
    ) -> List:
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

        Returns
        ----------
        list
            A list of datasets information.

        Example
        -------
        >>> dataset.list_datasets()
        """
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
        return response.json().get("Results")

    def download(
        self,
        group: str,
        dataset_hash: str,
        path: Optional[str] = "./",
        filename: Optional[str] = "dataset",
    ) -> None:
        """
        Download a dataset from mlops. The dataset will be a csv or parquet file.

        Parameters
        ----------
        group: Optional[str], optional
            Name of the group
        dataset_hash: Optional[str], optional
            Dataset hash
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

        url = f"{self.base_url}/datasets/result/{group}/{dataset_hash}"
        token = refresh_token(*self.credentials, self.base_url)
        response = make_request(
            url=url,
            method="GET",
            success_code=200,
            custom_exception=DatasetNotFoundError,
            custom_exception_message=f"Dataset not found.",
            specific_error_code=404,
            logger_msg=f"Unable to download dataset {dataset_hash}",
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


class MLOpsDataset(BaseModel):
    """
    Dataset class to represent mlops dataset.

    Parameters
    ----------
    login: str
        Login for authenticating with the client.
    password: str
        Password for authenticating with the client.
    url: str
        URL to MLOps Server. Default value is https://neomaril.datarisk.net, use it to test your deployment first before changing to production. You can also use the env variable MLOPS_URL to set this
    dataset_hash: str
        Dataset hash to download.
    dataset_name: str
        Name of the dataset.
    group: str
        Name of the group where we will search the dataset
    origin: str
        Origin of the dataset. It can be "Training", "Preprocessing", "Datasource" or "Model"
    """

    login: str = Field(exclude=True, repr=False)
    password: str = Field(exclude=True, repr=False)
    url: str = Field(exclude=True, repr=False)
    dataset_hash: str
    dataset_name: str
    group: str
    origin: str
    _client: MLOpsDatasetClient = PrivateAttr(
        None, init=False
    )  # Create new clients, for each operation

    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context: Any) -> None:
        if self._client is None:
            self._client = MLOpsDatasetClient(
                login=self.login,
                password=self.password,
                url=self.url,
            )

    def download(self, *, path: str = "./", filename: str = "dataset") -> None:
        """
        Download a dataset from mlops. The dataset will be a csv or parquet file.

        Parameters
        ---------
        path: str, optional
            Path to the downloaded dataset. Defaults to './'.
        filename: str, optional
            Name of the downloaded dataset. Defaults to 'dataset.parquet' or 'dataset.csv'.
        """
        self._client.download(
            group=self.group,
            dataset_hash=self.dataset_hash,
            path=path,
            filename=filename,
        )

    def train(self):
        raise NotImplementedError("Feature not implemented.")

    def preprocess(self):
        raise NotImplementedError("Feature not implemented.")

    def run_model(self):
        raise NotImplementedError("Feature not implemented.")
