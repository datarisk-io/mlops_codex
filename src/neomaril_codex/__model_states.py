"""
This file model states from different parts of the code. In the future, I must change the file name
to 'state.py' or something similar. The purpose will be modeling the possible states of each part of the
application
"""

from enum import Enum, auto


class ModelState(Enum):
    """
    Possible states of a mode
    """

    Ready = auto()
    Building = auto()
    Recovering = auto()
    FailedRecovery = auto()
    Failed = auto()
    Deployed = auto()
    Disabled = auto()
    DisabledRecovery = auto()
    DisabledFailed = auto()
    Deleted = auto()

    def __eq__(self, other):
        if isinstance(other, str):
            return str(self.name) == other
        return super().__eq__(other)

    def __str__(self):
        return self.name
