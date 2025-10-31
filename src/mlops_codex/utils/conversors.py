from pathlib import Path


def file_or_dataset(
    input_data: str,
    files: list,
    data: dict,
    path_field: str = None,
    dataset_field: str = None,
) -> None:
    """
    Check whether the given input data is a valid file path or dataset.
    If the input_data is a file, it performs file operations on the input and modifies the files list.
    Otherwise, it performs file operations on the data dictionary.
    Once the modifications are in place, there is no need to return a value.

    Args:
        input_data (str): Input data
        files (list): List of file paths
        data (dict): Dictionary of input data
        path_field (str): Field of the input data
        dataset_field (str): Field of the input data
    """
    path = Path(input_data)

    if path.exists() and path.is_file():
        files.append((path_field, (path.name, open(path, 'rb'))))
    else:
        data[dataset_field] = input_data
