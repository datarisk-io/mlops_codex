from pydantic import BaseModel, Field


class NeomarilAsyncPreprocessingV1(BaseModel):
    """
    A created `syncPreprocessing`.

    Attributes:
        hash (str): Neomaril unique identifier.
        group_name (str): Neomaril group name.
        status (str): The status of a `NeomarilAsyncPreprocessingV1`.
        python_version (str): The `python_version` of a `NeomarilAsyncPreprocessingV1`.
    """

    hash: str = Field(description="Unique hash identifier", alias="Hash")
    group_name: str = Field(description="Neomaril group", alias="Group")
    status: str = Field(description="Preprocessing status", alias="Status")
    python_version: str = Field(description="Python version", alias="PythonVersion")
