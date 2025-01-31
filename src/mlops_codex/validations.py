from datetime import datetime, timedelta
from typing import Tuple

from mlops_codex.exceptions import GroupError, InputError
from mlops_codex.base import BaseMLOpsClient


def validate_group_existence(group: str, client_object: BaseMLOpsClient) -> bool:
    """
    Validates that the given group exists.

    Args:
        group (str): The name of the group.
        client_object (BaseMLOpsClient): The client object that will be used to

    Returns:
        bool: True if the group exists

    Raises:
        GroupError: If the group does not exist
    """
    groups = [g.get("Name") for g in client_object.list_groups()]
    if group in groups:
        return True
    raise GroupError("Group dont exist. Create a group first.")


def validate_python_version(python_version: str) -> bool:
    """
    Validates that the Python version is valid.

    Args:
        python_version (str): The Python version to validate.

    Returns:
        bool: True if the Python version is valid, False otherwise.
    """
    if python_version not in ["3.8", "3.9", "3.10"]:
        raise InputError(
            "Invalid python version. Available versions are 3.8, 3.9, 3.10"
        )
    return True


def date_validation(start: str, end: str) -> Tuple[str, str]:
    if not start and not end:
        end = datetime.today().strftime("%d-%m-%Y")
        start = (datetime.today() - timedelta(days=6)).strftime("%d-%m-%Y")

    if not start and end:
        start = (datetime.strptime(end, "%d-%m-%Y") - timedelta(days=6)).strftime(
            "%d-%m-%Y"
        )

    if start and not end:
        end = (datetime.strptime(start, "%d-%m-%Y") + timedelta(days=6)).strftime(
            "%d-%m-%Y"
        )

    return start, end
