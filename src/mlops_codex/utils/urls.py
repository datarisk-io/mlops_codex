from enum import StrEnum

from mlops_codex.constants import BASE_URL


class AdminUrl(StrEnum):
    LOGIN_URL = f'{BASE_URL}/login'
    LIST_GROUP_URL = f'{BASE_URL}/groups'
    CREATE_GROUP_URL = f'{BASE_URL}/groups'
    REFRESH_GROUP_TOKEN_URL = f'{BASE_URL}/groups/refresh/{{group_name}}'


class TrainingUrl(StrEnum):
    REGISTER_URL = f'{BASE_URL}/training/register/{{group_name}}'
    UPLOAD_URL = f'{BASE_URL}/training/upload/{{group_name}}/{{training_hash}}'
    EXECUTE_URL = f'{BASE_URL}/training/execute/{{group_name}}/{{training_hash}}/{{execution_id}}'
    STATUS_URL = f'{BASE_URL}/training/status/{{group_name}}/{{execution_id}}'