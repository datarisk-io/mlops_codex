import pathlib

from mlops_codex.options.options import PythonVersion
from mlops_codex.utils.conversors import file_or_dataset


def assemble_custom_request_content(
    training_reference: str,
    run_name: str,
    python_version: PythonVersion,
    input_data: str,
    source: pathlib.Path,
    requirements: pathlib.Path,
    env_file: pathlib.Path = None,
    extras: list[pathlib.Path] = None,
):
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
