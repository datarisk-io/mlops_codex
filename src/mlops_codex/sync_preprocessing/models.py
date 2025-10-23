from typing import Any, Annotated

from pydantic import BaseModel, Field, AfterValidator

from pathlib import Path


def is_valid_filepath(path: Path) -> bool:
    path.exists()


class SyncPreprocessing(BaseModel):
    """
    A representation for `syncPreprocessings`.

    Attributes:
        name (str): human identifier.
        python_version (str): a Neomaril supported python version.
        script_reference (str): name of the function to be called.
        group_name (str): Neomaril group name.
        source_file_path (str): path to a valid file.
        requirements_file_path (str): path to a valid file.
        schema_file_path (str): path to a valid file.
    """

    name: str = Field(description="Preprocessing human identifier.")

    python_version: str = Field(description="A supported Neomaril python version.")

    script_reference: str = Field(description="Name of the function to be called.")

    group_name: str = Field(description="Neomaril group.")

    source_file_path: Path = Annotated[
        Path,
        Field(description="Path to the source file."),
        AfterValidator(is_valid_filepath),
    ]

    requirements_file_path: Path = Annotated[
        Path,
        Field(description="Path to the requirements file."),
        AfterValidator(is_valid_filepath),
    ]

    schema_file_path: Path = Annotated[
        Path,
        Field(description="Path to the schema file."),
        AfterValidator(is_valid_filepath),
    ]


class NeomarilSyncPreprocessing(BaseModel):
    """
    A created `syncPreprocessing`.

    Attributes:
        hash (str): Neomaril unique identifier.
        group_name (str): Neomaril group name.
    """

    hash: str = Field(description="Unique hash identifier", alias="Hash")
    group_name: str = Field(description="Neomaril group", alias="Group")
    status: str = Field(description="Preprocessing status", alias="Status")
    python_version: str = Field(description="Python version", alias="PythonVersion")
