from enum import StrEnum


class ExecutionStatus(StrEnum):
    UPLOADED = 'Uploaded'
    REQUESTED = 'Requested'
    RUNNING = 'Running'
    SUCCESS = "Succeeded"
    FAILED = "Failed"
