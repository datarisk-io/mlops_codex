from mlops_codex.logger_config import get_logger
from mlops_codex.shared.data_transmitter import send_file, send_json
from mlops_codex.shared.utils import parse_data
from mlops_codex.training.commons import register_execution
from mlops_codex.validations import validate_python_version

logger = get_logger()


def trigger_custom_training(**kwargs: dict) -> int:
    """
    Triggers a custom training execution.

    Parameters
    ----------
    **kwargs : dict
        Dictionary containing:
        - url : str
            Base URL for the API
        - token : str
            Authentication token
        - training_hash : str
            Training hash identifier
        - run_name : str
            Name of the run
        - description : str
            Description of the training
        - input_data : dict
            Input data dictionary
        - upload_data : dict
            Upload data dictionary
        - requirements_file : str
            Path to requirements file
        - source_file : str
            Path to source file
        - training_reference : str
            Training reference
        - python_version : str
            Python version to use
        - extra_files : list, optional
            List of tuples containing (name, path) for extra files
        - env : str, optional
            Path to env file

    Returns
    -------
    int
        Execution ID
    """

    execution_id = (
        kwargs.get("execution_id")
        if kwargs.get("execution_id", None) is not None
        else register_execution(
            url=f"{kwargs['url']}/v2/training/{kwargs['training_hash']}/execution",
            token=kwargs["token"],
            run_name=kwargs["run_name"],
            description=kwargs["description"],
            training_type="Custom",
        )
    )

    if kwargs["input_data"] is not None and kwargs["upload_data"] is not None:
        send_file(
            url=f"{kwargs['url']}/v2/training/execution/{execution_id}/input/file",
            token=kwargs["token"],
            input_data=kwargs["input_data"],
            upload_data=kwargs["upload_data"],
            neomaril_method="Upload input",
        )

    if kwargs["requirements_file"] is not None:
        send_file(
            url=f"{kwargs['url']}/v2/training/execution/{execution_id}/requirements-file",
            token=kwargs["token"],
            upload_data={"requirements": open(kwargs["requirements_file"], "rb")},
            neomaril_method="Upload requirements file",
        )

    if all(
        kwargs[key] for key in ["source_file", "training_reference", "python_version"]
    ):
        send_file(
            url=f"{kwargs['url']}/v2/training/execution/{execution_id}/script-file",
            token=kwargs["token"],
            input_data={
                "training_reference": kwargs["training_reference"],
                "python_version": validate_python_version(kwargs["python_version"]),
            },
            upload_data={"script": open(kwargs["source_file"], "rb")},
            neomaril_method="Upload script",
        )

    for name, path in kwargs.get("extra_files", []):
        send_file(
            url=f"{kwargs['url']}/v2/training/execution/{execution_id}/extra-file",
            token=kwargs["token"],
            input_data={"file_name": name},
            upload_data={"extra": open(path, "rb")},
            neomaril_method="Upload extra file",
        )

    if kwargs["env"] is not None:
        send_file(
            url=f"{kwargs['url']}/v2/training/execution/{execution_id}/env/file",
            token=kwargs["token"],
            upload_data={"env": open(kwargs["env"], "rb")},
            neomaril_method="Upload env",
        )

    return execution_id


def trigger_automl_training(**kwargs) -> int:
    """
    Triggers an AutoML training execution.

    Parameters
    ----------
    **kwargs : dict
        Dictionary containing:
        - url : str
            Base URL for the API
        - token : str
            Authentication token
        - training_hash : str
            Training hash identifier
        - run_name : str
            Name of the run
        - description : str
            Description of the training
        - conf_dict : str
            Path to configuration dictionary
        - input_data : dict
            Input data dictionary
        - upload_data : dict
            Upload data dictionary
        - extra_files : list, optional
            List of tuples containing (name, path) for extra files

    Returns
    -------
    int
        Execution ID
    """
    execution_id = (
        kwargs.get("execution_id")
        if kwargs.get("execution_id", None) is not None
        else register_execution(
            url=f"{kwargs['url']}/v2/training/{kwargs['training_hash']}/execution",
            token=kwargs["token"],
            run_name=kwargs["run_name"],
            description=kwargs["description"],
            training_type="AutoML",
        )
    )

    if kwargs["conf_dict"] is not None:
        send_file(
            url=f"{kwargs['url']}/v2/training/execution/{execution_id}/conf-dict/file",
            token=kwargs["token"],
            input_data=None,
            upload_data={"conf_dict": open(kwargs["conf_dict"], "rb")},
            neomaril_method="Upload conf_dict",
        )

    if kwargs["input_data"] is not None and kwargs["upload_data"] is not None:
        send_file(
            url=f"{kwargs['url']}/v2/training/execution/{execution_id}/input/file",
            token=kwargs["token"],
            input_data=kwargs["input_data"],
            upload_data=kwargs["upload_data"],
            neomaril_method="Upload input",
        )

    for name, path in kwargs.get("extra_files", []):
        send_file(
            url=f"{kwargs['url']}/v2/training/execution/{execution_id}/extra-file",
            token=kwargs["token"],
            input_data={"file_name": name},
            upload_data={"extra": open(path, "rb")},
            neomaril_method="Upload extra file",
        )

    return execution_id


def trigger_external_training(**kwargs) -> int:
    """
    Triggers an external training execution.

    Parameters
    ----------
    **kwargs : dict
        Dictionary containing:
        - url : str
            Base URL for the API
        - token : str
            Authentication token
        - training_hash : str
            Training hash identifier
        - data : dict
            Dictionary containing features, target and output file paths and hashes
        - run_name : str
            Name of the run
        - description : str
            Description of the training
        - python_version : str
            Python version to use
        - metrics_file : str, optional
            Path to metrics file
        - parameters_file : str, optional
            Path to parameters file
        - model_file : str, optional
            Path to model file
        - requirements_file : str, optional
            Path to requirements file

    Returns
    -------
    int
        Execution ID
    """
    execution_id = (
        kwargs.get("execution_id")
        if kwargs.get("execution_id", None) is not None
        else register_execution(
            url=f"{kwargs['url']}/v2/training/{kwargs['training_hash']}/execution",
            token=kwargs["token"],
            run_name=kwargs["run_name"],
            description=kwargs["description"],
            training_type="External",
        )
    )

    for var in ["features", "target", "output"]:
        file_path = kwargs.get(f"{var}_file")
        dataset_hash = kwargs.get(f"{var}_hash")

        if file_path is None and dataset_hash is None:
            continue
        
        form_data, upload_data = parse_data(
            file_path=file_path,
            form_data="dataset_name" if file_path else "dataset_hash",
            file_name=var if file_path else dataset_hash,
            file_form=var if file_path else None,
            dataset_hash=dataset_hash,
        )

        send_file(
            url=f"{kwargs['url']}/v2/training/execution/{execution_id}/{var}/file",
            token=kwargs["token"],
            input_data=form_data,
            upload_data=upload_data,
            neomaril_method=f"Upload {var} file",
        )

    if kwargs["metrics_file"] is not None:
        send_file(
            url=f"{kwargs['url']}/v2/training/execution/{execution_id}/metrics/file",
            token=kwargs["token"],
            upload_data={"metrics": open(kwargs["metrics_file"], "rb")},
            neomaril_method="Upload metrics file",
        )

    if kwargs["parameters_file"] is not None:
        send_file(
            url=f"{kwargs['url']}/v2/training/execution/{execution_id}/parameters/file",
            token=kwargs["token"],
            upload_data={"parameters": open(kwargs["parameters_file"], "rb")},
            neomaril_method="Upload parameters file",
        )

    if kwargs["model_file"] is not None:
        send_file(
            url=f"{kwargs['url']}/v2/training/execution/{execution_id}/model/file",
            token=kwargs["token"],
            upload_data={"model": open(kwargs["model_file"], "rb")},
            neomaril_method="Upload model file",
        )

    if kwargs["requirements_file"] is not None:
        send_file(
            url=f"{kwargs['url']}/v2/training/execution/{execution_id}/requirements-file",
            token=kwargs["token"],
            upload_data={"requirements": open(kwargs["requirements_file"], "rb")},
            neomaril_method="Upload requirements file",
        )

    if kwargs["python_version"] is not None:
        send_json(
            url=f"{kwargs['url']}/v2/training/execution/{execution_id}/python-version",
            token=kwargs["token"],
            input_data={"PythonVersion": validate_python_version(python_version=kwargs["python_version"])},
            neomaril_method="Set python version",
        )

    return execution_id
