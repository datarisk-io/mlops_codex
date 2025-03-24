from mlops_codex.http_request_handler import make_request
from mlops_codex.logger_config import get_logger

logger = get_logger()


def send_file(url, token, neomaril_method, input_data=None, upload_data=None):
    """
    Sends the file to the API.

    Parameters
    ----------
    url : str
        Base URL to send the file
    token : str
        Authorization token
    input_data: dict
        Dictionary containing the payload data
    upload_data : dict
        Dictionary containing the file data
    neomaril_method : str
        Method name for the Neomaril header
    """

    response = make_request(
        url=url,
        method="PATCH",
        success_code=201,
        data=input_data,
        files=upload_data,
        headers={
            "Authorization": f"Bearer {token}",
            "Neomaril-Origin": "Codex",
            "Neomaril-Method": neomaril_method,
        },
    ).json()

    msg = response["Message"]

    logger.info(msg)

    if "DatasetHash" in response:
        d_hash = response["DatasetHash"]
        logger.info(
            f"Dataset hash = {d_hash}",
        )


def send_json(url, token, payload, neomaril_method):
    """
    Sends the JSON payload to the API.

    Parameters
    ----------
    url : str
        URL to send the JSON payload
    token : str
        Authorization token
    payload : dict
        Dictionary containing the payload data
    neomaril_method : str
        Method name for the Neomaril header
    """

    response = make_request(
        url=url,
        method="PATCH",
        success_code=201,
        json=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Neomaril-Origin": "Codex",
            "Neomaril-Method": neomaril_method,
        },
    ).json()

    msg = response["Message"]

    logger.info({msg})
