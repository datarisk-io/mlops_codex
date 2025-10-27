from pathlib import Path

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
