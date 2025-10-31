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


def assemble_external_training_request_content(
    run_name: str,
    python_version: str,
    features: pathlib.Path,
    target: pathlib.Path,
    output: pathlib.Path,
    metrics: pathlib.Path = None,
    model: pathlib.Path = None,
    requirements: pathlib.Path = None,
    parameters: pathlib.Path = None,
    model_hash: str = None,
):
    """
    Assembles external training request content

    Args:
        run_name (str): Experiment name
        python_version (str): Python version. Available versions are 3.8, 3.9 and 3.10
        features (pathlib.Path): Input features used to train the model, needs to be a .parquet
        target (pathlib.Path): A .parquet file with the targets used to train
        output (pathlib.Path): A .parquet file with the predictions returned from the trained model
        metrics (pathlib.Path | None): A .json file with training metrics of the trained model
        model (pathlib.Path | None): A binary file with the model trained to be executed at the API. Allowed extensions are: .pkl, .pickle, .cbm, .json, .txt and .h5
        requirements (pathlib.Path | None): A .txt file with the packages used in the model
        parameters (pathlib.Path | None): A .json file containing experiment parameters
        model_hash (str | None): .json file containing experiment parameters

    Returns:
        (tuple[dict, list]): Return a tuple with the data and the files that will be uploaded
    """

    data = {
        'run_name': run_name,
        'training_type': 'External',
        'python_version': python_version,
    }

    files = [
        ('features', (features.name, open(features, 'rb'))),
        ('target', (target.name, open(target, 'rb'))),
        ('output', (output.name, open(output, 'rb'))),
    ]

    if metrics is not None:
        files.append(('metrics', (metrics.name, open(metrics, 'rb'))))

    if model is not None:
        files.append(('model', (model.name, open(model, 'rb'))))

    if requirements is not None:
        files.append(('requirements', (requirements.name, open(requirements, 'rb'))))

    if parameters is not None:
        files.append(('parameters', (parameters.name, open(parameters, 'rb'))))

    if model_hash is not None:
        data['model_hash'] = model_hash

    return data, files
