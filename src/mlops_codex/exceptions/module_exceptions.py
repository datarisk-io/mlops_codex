class TrainExecutionException(Exception):
    """Raised when an error occurs when executing a training an experiment"""

    def __init__(
        self,
        message: str = 'An error occurred when executing a training an experiment. '
        'Please, check the logs',
    ):
        super().__init__(message)
        self.message = message


class PythonVersionException(Exception):
    """Raised when an error occurs when executing a python version"""

    pass
