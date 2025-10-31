from enum import StrEnum


class ExecutionStatus(StrEnum):
    """
    Status of an execution (training, model execution, preprocessing execution)
    """

    UPLOADED = 'Uploaded'
    REQUESTED = 'Requested'
    RUNNING = 'Running'
    SUCCEEDED = 'Succeeded'
    FAILED = 'Failed'
