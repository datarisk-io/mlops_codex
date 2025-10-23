from pathlib import Path


def file_or_dataset(
        input_data: str,
        files: list,
        data: dict,
        path_field: str = None,
        dataset_field: str = None
) -> None:
    path = Path(input_data)

    if path.exists() and path.is_file():
        files.append((path_field, (path.name, open(path, 'rb'))))
    else:
        data[dataset_field] = input_data