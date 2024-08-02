from enum import Enum


class ModelState(Enum):
    """
    Possible states of a mode
    """

    Ready = 1
    Building = 2
    Recovering = 3
    FailedRecovery = 4
    Failed = 5
    Deployed = 6
    Disabled = 7
    DisabledRecovery = 8
    DisabledFailed = 9
    Deleted = 10

    def __eq__(self, other):
        if isinstance(other, str):
            return str(self.name) == other
        return super().__eq__(other)

    def __str__(self):
        return self.name


k = ModelState['Ready']
print(f"Model is {k}")

print(k in [ModelState.Ready, ModelState.Building])
print(k == 'Ready')

