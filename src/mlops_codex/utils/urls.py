from enum import StrEnum

from mlops_codex.constants import BASE_URL


class AdminUrl(StrEnum):
    """
    Url for administration endpoints
    """
    LOGIN_URL = f'{BASE_URL}/login'
    LIST_GROUP_URL = f'{BASE_URL}/groups'
    CREATE_GROUP_URL = f'{BASE_URL}/groups'
    REFRESH_GROUP_TOKEN_URL = f'{BASE_URL}/groups/refresh/{{group_name}}'


class TrainingUrl(StrEnum):
    """
    Url for training endpoints
    """
    REGISTER_URL = f'{BASE_URL}/training/register/{{group_name}}'
    UPLOAD_URL = f'{BASE_URL}/training/upload/{{group_name}}/{{training_hash}}'
    EXECUTE_URL = (
        f'{BASE_URL}/training/execute/{{group_name}}/{{training_hash}}/{{execution_id}}'
    )
    STATUS_URL = f'{BASE_URL}/training/status/{{group_name}}/{{execution_id}}'
    PROMOTE_URL = (
        f'{BASE_URL}/training/promote/{{group_name}}/{{training_hash}}/{{execution_id}}'
    )


class SyncPreprocessingUrl(StrEnum):
    """
    Url for SyncPreprocessing endpoints
    """
    REGISTER_URL = f'{BASE_URL}/preprocessing/register/{{group_name}}'
    STATUS_HOST_URL = f'{BASE_URL}/preprocessing/status/{{group_name}}/{{script_hash}}'
    LOGS_URL = f'{BASE_URL}/preprocessing/logs/{{group_name}}/{{script_hash}}'
    DESCRIBE_URL = f'{BASE_URL}/preprocessing/describe/{{group_name}}/{{script_hash}}'
    HOST_URL = f'{BASE_URL}/preprocessing/sync/host/{{group_name}}/{{script_hash}}'
    SEARCH_URL = f'{BASE_URL}/preprocessing/search'
    RUN_URL = f'{BASE_URL}/preprocessing/sync/run/{{group_name}}/{{script_hash}}'


class AsyncPreprocessingUrlV1(StrEnum):
    """
    Url for AsyncPreprocessing endpoints
    """
    STATUS_HOST_URL = f'{BASE_URL}/preprocessing/status/{{group_name}}/{{script_hash}}'
    LOGS_URL = f'{BASE_URL}/preprocessing/logs/{{group_name}}/{{script_hash}}'
    DESCRIBE_URL = f'{BASE_URL}/preprocessing/describe/{{group_name}}/{{script_hash}}'
    HOST_URL = f'{BASE_URL}/preprocessing/async/host/{{group_name}}/{{script_hash}}'
    SEARCH_URL = f'{BASE_URL}/preprocessing/search'
    RUN_URL = f'{BASE_URL}/preprocessing/async/run/{{group_name}}/{{script_hash}}'
    STATUS_EXECUTION_URL = f'{BASE_URL}/preprocessing/async/status/{{group_name}}/{{execution_id}}'
