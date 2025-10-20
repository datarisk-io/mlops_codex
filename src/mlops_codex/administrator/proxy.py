from functools import wraps

from attrs import define

from mlops_codex.administrator.admin import MLOpsAdmin


@define(slots=True)
class ServiceProxy:
    """
    The service proxy is intended to intercept every method and check if the user
    is logged in.
    """
    service: object
    admin: MLOpsAdmin

    def __getattr__(self, method_name):
        attr = getattr(self.service, method_name)

        if not callable(attr):
            raise AttributeError(f"Method '{method_name}' not found.")

        @wraps(attr)
        def wrapper(*args, **kwargs):
            headers = {'Authorization': f'Bearer {self.admin.token}'}
            kwargs['headers'] = headers
            return attr(*args, **kwargs)

        return wrapper
