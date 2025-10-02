import os
from http import HTTPStatus

from attrs import define, field
from cachetools.func import ttl_cache
from dotenv import load_dotenv, find_dotenv

from V2.base.client import send_http_request
from V2.utils.logger_configuration import get_logger
from V2.utils.urls import AdminUrl

load_dotenv(find_dotenv(usecwd=True))
logger = get_logger()


@define(slots=True)
class MLOpsAdmin:
    """
    Administration class for the MLOps API.
    """

    email = field(default=os.getenv('MLOPS_USER', None), type=str)
    password = field(default=os.getenv('MLOPS_PASSWORD', None))
    tenant = field(default=os.getenv('MLOPS_TENANT', None))
    token = field(default=None, repr=False)

    @ttl_cache(ttl=10800)
    def _refresh_login_token(self) -> str:
        """
        Refresh login token

        Returns:
            (str): Return new login token
        """

        response = send_http_request(
            url=AdminUrl.LOGIN_URL,
            method='POST',
            successful_code=HTTPStatus.OK,
            data={"user": self.email, "password": self.password, "tenant": self.tenant}
        )

        return response.json()['Token']


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
                "Authorization": "Bearer " + self.token,
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.list_groups.__qualname__,
            },
        )

        return response.json()["Results"]

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
            data={"name": name, "description": description},
            headers={
                "Authorization": "Bearer " + self.token,
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.create_group.__qualname__,
            },
        )

        group_token = response.json()['Token']
        logger.info(
            f"Group '{name}' inserted. Carefully save it as we won't show it again. "
            f"Group token: {group_token}"
        )
        return group_token

    def refresh_group_token(self, name: str, force: bool = False) -> str:
        """
        Refresh the token of the group.

        Args:
            name (str): The name of the group to refresh the token.
            force (bool, optional): Whether to force refresh of the token. If token is still valid, it won't be changed,
                                    unless force is True, the token will not be changed. Defaults to False.

        Returns:
            (str): New group token.
        """
        response = send_http_request(
            url=AdminUrl.REFRESH_GROUP_TOKEN_URL.format(group_name=name),
            method='GET',
            successful_code=HTTPStatus.CREATED,
            params={"force": str(force).lower()},
            headers={
                "Authorization": "Bearer " + self.token,
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.refresh_group_token.__qualname__,
            }
        )

        group_token = response.json()['Token']
        logger.info(f"Group '{name}' has had its token refreshed.")
        return group_token
