import pathlib

from mlops_codex.utils.conversors import file_or_dataset


def assemble_custom_request_content(
    training_reference: str,
    run_name: str,
    python_version: str,
    input_data: str,
    source: pathlib.Path,
    requirements: pathlib.Path,
    env_file: pathlib.Path = None,
    extras: list[pathlib.Path] = None,
):
    """
    Assembles custom training request content

    Args:
        training_reference (str): Entrypoint function name
        run_name (str): Experiment name
        python_version (str): Python version. Available versions are 3.8, 3.9 and 3.10
        input_data (str): Input data. It can be a path to a file or a dataset hash which a string
        source (pathlib.Path): Path to the .py script with an entry point function
        requirements (pathlib.Path): Path to the requirements file. It must be a txt file
        env_file (pathlib.Path): Path to the .env file
        extras (list[pathlib.Path]): Paths to extras file. It can be a list of extra files

    Returns:
        (tuple[dict, list]): Return a tuple with the data and the files that will be uploaded
    """
    data = {
        'training_reference': training_reference,
        'run_name': run_name,
        'python_version': python_version,
        'training_type': 'Custom',
    }

    files = [
        ('source', (source.name, open(source, 'rb'))),
        ('requirements', (requirements.name, open(requirements, 'rb'))),
    ]

    file_or_dataset(
        input_data=input_data,
        files=files,
        data=data,
        path_field='train_data',
        dataset_field='dataset_hash',
    )

    if env_file is not None:
        files.append(('env', (env_file.name, open(env_file, 'rb'))))

    if extras is not None:
        extra_data = [('extra', (e.name, open(e, 'rb'))) for e in extras]
        files += extra_data

    return data, files


def assemble_automl_request_content(
    run_name: str,
    input_data: str,
    configuration: pathlib.Path,
):
    """
    Assembles automl request content

    Args:
        run_name (str): Experiment name
        input_data (str): Input data. It can be a path to a file or a dataset hash which a string
        configuration (pathlib.Path): Path to the configuration file. It must be a json file

    Returns:
        (tuple[dict, list]): Return a tuple with the data and the files that will be uploaded
    """

    data = {
        'run_name': run_name,
        'training_type': 'AutoML',
    }

    files = [
        ('conf_dict', (configuration.name, open(configuration, 'rb'))),
    ]

    file_or_dataset(
        input_data=input_data,
        files=files,
        data=data,
        path_field='train_data',
        dataset_field='dataset_hash',
    )

    return data, files
