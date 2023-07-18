class AuthenticationError(Exception):
    """Raised when authentication fails (401 response from server)"""
    pass

class ServerError(Exception):
    """Raised when server returns a 500 response"""
    pass

class ModelError(Exception):
    """Raised when a model is not avaliable"""
    pass

class PreprocessingError(Exception):
    """Raised when a pre processing is not avaliable"""
    pass

class GroupError(Exception):
    """Raised when a group is not avaliable"""
    pass

class ExecutionError(Exception):
    """Raised when a execution is not avaliable"""
    pass

class TrainingError(Exception):
    """Raised when a training is not avaliable"""
    pass

class InputError(Exception):
    """Raised when a user input is not valid"""
    pass

class PipelineError(Exception):
    """Raised when the pipeline configuration is invalid"""
    pass