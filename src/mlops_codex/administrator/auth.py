from http import HTTPStatus

from cachetools.func import ttl_cache
from pydantic import BaseModel, EmailStr, Field

from mlops_codex.base.client import send_http_request
from mlops_codex.utils.urls import AdminUrl


@ttl_cache(ttl=10800)
def _login(email: EmailStr, password: str, tenant: str) -> str:
    """
    Refresh login token

    Args:
        email (EmailStr): Email address of the user.
        password (str): Password of the user.
        tenant (str): Tenant where user is connected.

    Returns:
        (str): Return new login token
    """

    response = send_http_request(
        url=AdminUrl.LOGIN_URL,
        method='POST',
        successful_code=HTTPStatus.OK,
        data={'user': email, 'password': password, 'tenant': tenant},
    )

    return response.json()['Token']


class AuthManager(BaseModel):
    """
    Class responsible for login and authentication the user.

    Args:
        email (EmailStr): Email address of the user.
        password (str): Password of the user.
        tenant (str): Tenant where user is connected.
    """

    email: EmailStr = Field(alias='email')
    password: str = Field(repr=False)
    tenant: str = Field(alias='tenant')

    @property
    def header(self) -> dict[str, str]:
        """
        Return login header

        Returns:
            (dict): Return login header to inject into API calls.
        """
        token = _login(self.email, self.password, self.tenant)
        return {'Authorization': f'Bearer {token}'}
