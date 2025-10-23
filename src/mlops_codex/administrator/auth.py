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
        email (EmailStr): Email address
        password (str): Password
        tenant (str): Tenant

    Returns:
        (str): Return new login token
    """

    print("Admin url", AdminUrl.LOGIN_URL)

    response = send_http_request(
        url=AdminUrl.LOGIN_URL,
        method='POST',
        successful_code=HTTPStatus.OK,
        data={'user': email, 'password': password, 'tenant': tenant},
    )

    return response.json()['Token']


class AuthManager(BaseModel):

    email: EmailStr = Field(alias='email')
    password: str = Field(repr=False)
    tenant: str = Field(alias='tenant')

    @property
    def token(self) -> str:
        return _login(self.email, self.password, self.tenant)