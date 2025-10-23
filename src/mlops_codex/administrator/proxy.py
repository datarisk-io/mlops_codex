from functools import wraps

from mlops_codex.administrator.auth import AuthManager


class ServiceProxy:
    """
    The service proxy is intended to intercept every method and check if the user
    is logged in.
    """

    def __init__(self, service: object, auth_manager: AuthManager):
        self.__service = service
        self.__auth_manager = auth_manager

    def __getattr__(self, method_name):
        attr = getattr(self.__service, method_name)

        if not callable(attr):
            raise AttributeError(f"Method '{method_name}' not found.")

        @wraps(attr)
        def wrapper(*args, **kwargs):
            headers = {
                'Authorization': f'Bearer {self.__auth_manager.token}',
                'Neomaril-Origin': 'Codex',
                'Neomaril-Method': method_name,
            }
            kwargs['headers'] = headers
            return attr(*args, **kwargs)

        return wrapper
