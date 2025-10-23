from http import HTTPStatus
from typing import Any, cast

from pydantic import BaseModel, EmailStr, Field, PrivateAttr

from mlops_codex.administrator.auth import AuthManager
from mlops_codex.administrator.proxy import ServiceProxy
from mlops_codex.base.client import send_http_request
from mlops_codex.logger_config import get_logger
from mlops_codex.train.client import MLOpsTrainClient
from mlops_codex.utils.urls import AdminUrl

logger = get_logger()


class Admin(BaseModel):
    """
    Administration class for the Datarisk MLOps API.
    """

    email: EmailStr | str = Field(alias='email')
    password: str = Field(repr=False)
    tenant: str = Field(alias='tenant')

    __train: ServiceProxy = PrivateAttr()

    __auth: AuthManager = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        self.__auth = AuthManager(email=self.email, password=self.password, tenant=self.tenant)
        self.__train = ServiceProxy(service=MLOpsTrainClient(), auth_manager=self.__auth)

    @property
    def train(self) -> MLOpsTrainClient:
        return cast(MLOpsTrainClient, cast(object, self.__train))

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
            headers={
                'Authorization': 'Bearer ' + self._login_token,
                'Neomaril-Origin': 'Codex',
                'Neomaril-Method': self.list_groups.__qualname__,
            },
        )

        return response.json()['Results']

    def create_group(self, name: str, description: str) -> str:
        """
        Create a group for multiple models of the same final client.
        Keep the token information to run the model of that group later.

        Args:
            name (str): The name of the group.
            description (str): The description of the group.

        Returns:
            str: The token of the created group.
        """
        response = send_http_request(
            url=AdminUrl.CREATE_GROUP_URL,
            method='POST',
            successful_code=HTTPStatus.CREATED,
            data={'name': name, 'description': description},
            headers={
                'Authorization': 'Bearer ' + self._login_token,
                'Neomaril-Origin': 'Codex',
                'Neomaril-Method': self.create_group.__qualname__,
            },
        )

        group_token = response.json()['Token']
        logger.info(
            f"Group '{name}' inserted. Carefully save it as we won't show it again. "
            f'Group token: {group_token}'
        )
        return group_token

    def refresh_group_token(self, name: str, force: bool = False) -> str:
        """
        Refresh the token of the group.

        Args:
            name (str): The name of the group to refresh the token.
            force (bool, optional): Whether to force refresh of the token.
                If token is still valid, it won't be changed, unless force is
                True, the token will not be changed. Defaults to False.

        Returns:
            (str): New group token.
        """
        response = send_http_request(
            url=AdminUrl.REFRESH_GROUP_TOKEN_URL.format(group_name=name),
            method='GET',
            successful_code=HTTPStatus.CREATED,
            params={'force': str(force).lower()},
            headers={
                'Authorization': 'Bearer ' + self._login_token,
                'Neomaril-Origin': 'Codex',
                'Neomaril-Method': self.refresh_group_token.__qualname__,
            },
        )

        group_token = response.json()['Token']
        logger.info(f"Group '{name}' has had its token refreshed.")
        return group_token
