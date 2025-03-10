"""
This file model states from different parts of the code. In the future, I must change the file name
to 'state.py' or something similar. The purpose will be modeling the possible states of each part of the
application
"""

from enum import Enum


class ModelTypes(str, Enum):

    Async = "Async"
    Sync = "Sync"

    def __str__(self):
        return self.name


class ModelState(str, Enum):
    """
    States of a model
    """

    Ready = "Ready"
    Building = "Building"
    Recovering = "Recovering"
    FailedRecovery = "FailedRecovery"
    Failed = "Failed"
    Deployed = "Deployed"
    Disabled = "Disabled"
    DisabledRecovery = "DisabledRecovery"
    DisabledFailed = "DisabledFailed"
    Deleted = "Deleted"

    def __str__(self):
        return self.name


class ModelExecutionState(str, Enum):
    """
    State of a model execution
    """

    Requested = "Requested"
    Running = "Running"
    Succeeded = "Succeeded"
    Failed = "Failed"

    def __str__(self):
        return self.name


class MonitoringStatus(str, Enum):
    """
    State of monitoring
    """

    Unvalidated = "Unvalidated"
    Validating = "Validating"
    Validated = "Validated"
    Invalidated = "Invalidated"

    def __str__(self):
        return self.name
