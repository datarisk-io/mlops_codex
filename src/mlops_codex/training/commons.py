from mlops_codex.http_request_handler import make_request
from mlops_codex.logger_config import get_logger

logger = get_logger()


def register_execution(url, token, run_name, description, training_type):
    """
    Register a training execution.

    Parameters
    ----------
    url: str
        URL to register the execution.
    token: str
        Authentication token.
    run_name: str
        Name of the run.
    description: str
        Description of the training.
    training_type: str
        Type of training.
    """
    register_training_response = make_request(
        url=url,
        method="POST",
        success_code=201,
        json={
            "RunName": run_name,
            "Description": description,
            "TrainingType": training_type,
        },
        headers={
            "Authorization": f"Bearer {token}",
            "Neomaril-Origin": "Codex",
            "Neomaril-Method": "Register execution",
        },
    ).json()

    msg = register_training_response.get("Message")
    execution_id = register_training_response["ExecutionId"]
    logger.info(f"{msg} for {run_name}")

    return execution_id
