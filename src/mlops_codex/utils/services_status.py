from enum import StrEnum


class ExecutionStatus(StrEnum):
    UPLOADED = 'Uploaded'
    REQUESTED = 'Requested'
    RUNNING = 'Running'
    SUCCEEDED = 'Succeeded'
    FAILED = 'Failed'
