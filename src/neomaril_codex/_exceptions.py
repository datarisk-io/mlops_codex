class AuthenticationError(Exception):
    """Raised when authentication fails (401 response from server)"""
    pass

class ServerError(Exception):
    """Raised when server returns a 500 response"""
    pass

class ModelError(Exception):
    """Raised when a model is not avaliable"""
    pass

class InputError(Exception):
    """Raised when a user input is not valid"""
    pass