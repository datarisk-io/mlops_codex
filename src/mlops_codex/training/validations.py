# Specific validators for training
# TODO: maybe it can be generic
from mlops_codex.exceptions import InputError


def validate_input(required, fields):
    if (not all(k in fields for k in required)) or (
        not all(fields[f] for f in required)
    ):
        raise InputError(
            f"The parameters {required} are mandatory for this training execution type."
        )

    return True
