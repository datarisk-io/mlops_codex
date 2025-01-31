from typing import Any, List, Optional

from pydantic import BaseModel, Field, PrivateAttr

from mlops_codex.base import BaseMLOpsClient
from mlops_codex.exceptions import DatasetNotFoundError
from mlops_codex.http_request_handler import make_request, refresh_token
from mlops_codex.logger_config import get_logger
from mlops_codex.preprocessing import MLOpsPreprocessingAsyncV2Client

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
        group: str
            Name of the group
        dataset_hash: str
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
            custom_exception_message="Dataset not found.",
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
    _client: MLOpsDatasetClient = PrivateAttr(None, init=False)

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

    def host_preprocessing(
        self,
        *,
        name: str,
        group: str,
        script_path: str,
        entrypoint_function_name: str,
        requirements_path: str,
        python_version: Optional[str] = "3.9",
    ):
        """
        Host a preprocessing script via dataset module. By default, the user will host and wait the hosting. It returns a MLOpsPreprocessingAsyncV2, then you can run it.

        Parameters
        ----------
        name: str
            Name of the new preprocessing script
        group: str
            Group of the new preprocessing script
            Dataset to upload schema to
        script_path: str
            Path to the python script
        entrypoint_function_name: str
            Name of the entrypoint function in the python script
        python_version: str
            Python version for the model environment. Available versions are 3.8, 3.9, 3.10. Defaults to '3.9'
        requirements_path: str
            Path to the requirements file

        Returns
        -------
        MLOpsPreprocessingAsyncV2
            Preprocessing async version of the new preprocessing script.
        """
        raise NotImplementedError("Host preprocessing not implemented.")

    def run_preprocess(
        self,
        *,
        preprocessing_script_hash: str,
        execution_id: int,
    ):
        """
        Run a preprocessing script execution from a dataset. By default, the user will run the preprocessing script and wait until it completes.

        Parameters
        ----------
        preprocessing_script_hash: str
            Hash of the preprocessing script
        execution_id: int
            Preprocessing Execution ID
        """
        raise NotImplementedError("Run preprocessing not implemented.")

    def train(self):
        raise NotImplementedError("Feature not implemented.")

    def run_model(self):
        raise NotImplementedError("Feature not implemented.")
