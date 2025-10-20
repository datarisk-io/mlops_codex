from enum import StrEnum

from mlops_codex.constants import BASE_URL


class AdminUrl(StrEnum):
    LOGIN_URL = f'{BASE_URL}/login'
    LIST_GROUP_URL = f'{BASE_URL}/groups'
    CREATE_GROUP_URL = f'{BASE_URL}/groups'
    REFRESH_GROUP_TOKEN_URL = f'{BASE_URL}/groups/refresh/{{group_name}}'
