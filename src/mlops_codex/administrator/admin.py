from http import HTTPStatus
from typing import Any

from pydantic import BaseModel, EmailStr, Field, PrivateAttr

from mlops_codex.administrator.auth import AuthManager
from mlops_codex.base.client import send_http_request
from mlops_codex.logger_config import get_logger
from mlops_codex.train.train import MLOpsTrainClient
from mlops_codex.utils.urls import AdminUrl

logger = get_logger()


class Admin(BaseModel):
    """
    Administration class for the Datarisk MLOps API.
    """

    email: EmailStr | str
    password: str = Field(repr=False)
    tenant: str

    __train: MLOpsTrainClient = PrivateAttr()

    __auth: AuthManager = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        self.__auth = AuthManager(
            email=self.email, password=self.password, tenant=self.tenant
        )
        self.__train = MLOpsTrainClient(auth=self.__auth)

    @property
    def train(self) -> MLOpsTrainClient:
        return self.__train

    def list_groups(self) -> list[str]:
        """
        List all MLOps groups the user has access to.

        Returns:
            list[str]: List of MLOps groups the user has access to.

        """
        response = send_http_request(
            url=AdminUrl.LIST_GROUP_URL,
            method='GET',
            successful_code=HTTPStatus.OK,
            headers=self.__auth.header,
        )

        return response.json()['Results']

    def create_group(self, name: str, description: str) -> str:
        """
        Create a group for multiple models of the same final client.
        Keep the header information to run the model of that group later.

        Args:
            name (str): The name of the group.
            description (str): The description of the group.

        Returns:
            str: The header of the created group.
        """
        response = send_http_request(
            url=AdminUrl.CREATE_GROUP_URL,
            method='POST',
            successful_code=HTTPStatus.CREATED,
            data={'name': name, 'description': description},
            headers=self.__auth.header,
        )

        group_token = response.json()['Token']
        logger.info(
            f"Group '{name}' inserted. Carefully save it as we won't show it again. "
            f'Group header: {group_token}'
        )
        return group_token

    def refresh_group_token(self, name: str, force: bool = False) -> str:
        """
        Refresh the header of the group.

        Args:
            name (str): The name of the group to refresh the header.
            force (bool, optional): Whether to force refresh of the header.
                If header is still valid, it won't be changed, unless force is
                True, the header will not be changed. Defaults to False.

        Returns:
            (str): New group header.
        """
        response = send_http_request(
            url=AdminUrl.REFRESH_GROUP_TOKEN_URL.format(group_name=name),
            method='GET',
            successful_code=HTTPStatus.CREATED,
            params={'force': str(force).lower()},
            headers=self.__auth.header,
        )

        group_token = response.json()['Token']
        logger.info(f"Group '{name}' has had its header refreshed.")
        return group_token
