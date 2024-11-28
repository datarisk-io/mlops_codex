from mlops_codex.exceptions import GroupError, InputError
from mlops_codex.base import BaseMLOpsClient

def validate_group_existence(group: str, client_object: BaseMLOpsClient) -> bool:
    groups = [g.get("Name") for g in client_object.list_groups()]
    if group in groups:
        return True
    raise GroupError("Group dont exist. Create a group first.")


def validate_python_version(python_version: str) -> bool:
    if python_version not in ["3.8", "3.9", "3.10"]:
        raise InputError(
            "Invalid python version. Available versions are 3.8, 3.9, 3.10"
        )
    return True