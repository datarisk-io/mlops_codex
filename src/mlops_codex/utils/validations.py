from pathlib import Path
from typing import Callable

from mlops_codex.exceptions.module_exceptions import PythonVersionException


def validate_python_version(python_version: str) -> str:
    """
    Validates that the Python version is valid.

    Args:
        python_version (str): The Python version to validate.

    Returns:
        bool: True if the Python version is valid, False otherwise.
    """
    if python_version not in ['3.8', '3.9', '3.10']:
        raise PythonVersionException(
            'Invalid python version. Available versions are 3.8, 3.9, 3.10'
        )
    return 'Python' + python_version.replace('.', '')


def str_to_path(path: str) -> Path | None:
    """
    Convert a string path to a Path object.

    Args:
        path (str): A string path.

    Returns:
        (Path): A Path object.
    """

    if path is None:
        return None

    path = Path(path)
    if path.exists():
        return path
    raise FileNotFoundError(f"Path '{path}' does not exist")


def file_extension_validation(*permitted_extensions: str) -> Callable[[Path], Path]:
    """
    Validates that the file extension is valid.

    Args:
        permitted_extensions: List of permitted file extensions.
    Returns:
        (bool): True if the file extension is valid, otherwise raises an exception.
    """

    def _validate(path: Path) -> Path:
        if path.suffix.lower() not in permitted_extensions:
            raise ValueError(
                f"File '{path.name}' must have extension {permitted_extensions}"
            )
        return path

    return _validate
