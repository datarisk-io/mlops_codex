from typing import BinaryIO, Dict, Optional, Tuple

import requests
from cachetools.func import ttl_cache

from mlops_codex.exceptions import InputError
from mlops_codex.shared.constants import CODEX_VERSION


def parse_data(
    file_path: Optional[str] = None,
    form_data: Optional[str] = None,
    file_name: Optional[str] = None,
    file_form: Optional[str] = None,
    dataset_hash: Optional[str] = None,
) -> Tuple[Dict[str, str], Optional[Dict[str, BinaryIO]]]:
    """
    Parses input data for file uploads, validating the combinations of parameters.

    Parameters
    ----------
    file_path : str, optional
        Path to the file to be uploaded
    form_data : str, optional
        Form data key
    file_name : str, optional
        Name of the file
    file_form : str, optional
        Form key for the file upload
    dataset_hash : str, optional
        Hash of the dataset

    Returns
    -------
    tuple
        A tuple containing:
        - Dict with form data (file name or dataset hash)
        - Dict with file data or None

    Raises
    ------
    InputError
        If an invalid combination of parameters is provided
    """
    if file_path is not None and dataset_hash is not None:
        raise InputError(
            "You must provide either a file path or a dataset hash, not both."
        )

    if file_path is not None:
        if any(param is None for param in [form_data, file_name, file_form]):
            raise InputError(
                "When providing a file path, form_data, file_name, and file_form are required."
            )
        return {form_data: file_name}, {file_form: open(file_path, "rb")}

    if dataset_hash is not None:
        if file_path is not None:
            raise InputError(
                "When providing a dataset hash, file_path must not be provided."
            )
        return {form_data: dataset_hash}, None

    raise InputError("You must provide either a file path or a dataset hash.")


@ttl_cache
def check_lib_version():
    response = requests.get(
        "https://pypi.org/pypi/datarisk-mlops-codex/json", timeout=60
    )

    if response.status_code != 200:
        return

    json_data = response.json()

    info = json_data["info"]
    major_version = info["version"]
    if major_version != CODEX_VERSION:
        print(
            f"You are using {CODEX_VERSION}, but version {major_version} is recommended."
        )
