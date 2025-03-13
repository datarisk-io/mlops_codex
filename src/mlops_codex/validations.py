from datetime import datetime, timedelta
from typing import Set, Tuple

from mlops_codex.base import BaseMLOpsClient
from mlops_codex.dataset import MLOpsDataset
from mlops_codex.exceptions import GroupError, InputError


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


def validate_python_version(python_version: str) -> str:
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
    return "Python" + python_version.replace(".", "")


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


def file_extension_validation(file_name: str, permitted_extensions: Set[str]):
    if file_name.split(".")[-1] not in permitted_extensions:
        raise InputError(f"File {file_name} must have extension {permitted_extensions}")
    return True


def validate_data(data, permitted_extensions: Set[str]):
    if isinstance(data, list) and isinstance(data[0], tuple):
        for name, path in data:
            file_extension_validation(path, permitted_extensions)
    elif isinstance(data, str):
        file_extension_validation(data, permitted_extensions)
    elif isinstance(data, tuple):
        name, path = data
        file_extension_validation(path, permitted_extensions)
    elif isinstance(data, MLOpsDataset) or isinstance(data[0], MLOpsDataset):
        return True
    else:
        raise InputError(
            "Invalid data type. Please provide a valid data type such as list | str | tuple | MLOpsDataset"
        )
