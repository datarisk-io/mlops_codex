from enum import StrEnum

from mlops_codex.constants import BASE_URL


class AdminUrl(StrEnum):
    LOGIN_URL = f'{BASE_URL}/login'
    LIST_GROUP_URL = f'{BASE_URL}/groups'
    CREATE_GROUP_URL = f'{BASE_URL}/groups'
    REFRESH_GROUP_TOKEN_URL = f'{BASE_URL}/groups/refresh/{{group_name}}'


class SyncPreprocessingUrl(StrEnum):
    REGISTER_URL = f'{BASE_URL}/preprocessing/register/{{group_name}}'
    STATUS_HOST_URL = f'{BASE_URL}/preprocessing/status/{{group_name}}/{{script_hash}}'
    LOGS_URL = f'{BASE_URL}/preprocessing/logs/{{group_name}}/{{script_hash}}'
    DESCRIBE_URL = f'{BASE_URL}/preprocessing/describe/{{group_name}}/{{script_hash}}'
    HOST_URL = f'{BASE_URL}/preprocessing/sync/host/{{group_name}}/{{script_hash}}'
    SEARCH_URL = f'{BASE_URL}/preprocessing/search'
    RUN_URL = f'{BASE_URL}/preprocessing/sync/run/{{group_name}}/{{script_hash}}'


class AsyncPreprocessingUrlV1(StrEnum):
    STATUS_HOST_URL = f'{BASE_URL}/preprocessing/status/{{group_name}}/{{script_hash}}'
    LOGS_URL = f'{BASE_URL}/preprocessing/logs/{{group_name}}/{{script_hash}}'
    DESCRIBE_URL = f'{BASE_URL}/preprocessing/describe/{{group_name}}/{{script_hash}}'
    HOST_URL = f'{BASE_URL}/preprocessing/async/host/{{group_name}}/{{script_hash}}'
    SEARCH_URL = f'{BASE_URL}/preprocessing/search'
    RUN_URL = f'{BASE_URL}/preprocessing/async/run/{{group_name}}/{{script_hash}}'
    STATUS_EXECUTION_URL = f'{BASE_URL}/preprocessing/async/status/{{group_name}}/{{execution_id}}'
