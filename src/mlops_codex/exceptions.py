class AuthenticationError(Exception):
    """Raised when authentication fails (401 response from server)"""

    pass


class ServerError(Exception):
    """Raised when server returns a 500 response"""

    pass


class ModelError(Exception):
    """Raised when a model is not available"""

    pass


class PreprocessingError(Exception):
    """Raised when a preprocessing is not available"""

    pass


class GroupError(Exception):
    """Raised when a group is not available"""

    pass


class ExecutionError(Exception):
    """Raised when a execution is not available"""

    pass


class TrainingError(Exception):
    """Raised when a training is not available"""

    pass


class InputError(Exception):
    """Raised when a user input is not valid"""

    pass


class PipelineError(Exception):
    """Raised when the pipeline configuration is invalid"""

    pass


class DataSorceError(Exception):
    """Raised when a datasource has a problem"""

    pass


class DatasetNotFoundError(Exception):
    """Raised when a datasource has a problem"""

    pass


class CredentialError(Exception):
    """Raised when the datasource cloud credential it's invalid"""

    pass


class ExternalMonitoringError(Exception):
    """Raised when an external monitoring is not available"""

    pass


class UnexpectedError(Exception):
    """Raised when an unexpected error occurs"""

    pass
