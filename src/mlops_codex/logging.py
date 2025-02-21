import json
import os
from typing import Union

from mlops_codex.exceptions import InputError


class Logger:
    """
    Class to custom logger for model scripts.

    Attributes
    -----------
    model_type: str
        Attribute that designates the type of the model being executed. Can be 'Sync' or 'Async'

    Raises
    -----------
    InputError
        Invalid input for the logging functions

    Example
    --------
    The logger needs to be implemented inside the function being executed by MLOps like this:

    .. code-block:: python

        from joblib import load
        import pandas as pd
        from mlops_codex.logging import Logger


        def score(data_path, model_path):
            logger = Logger('Async')

            logger.debug("USING LOGGER")

            model = load(model_path+"/model.pkl")

            df = pd.read_csv(data_path+'/input.csv')

            if len(df) < 5:
                logger.warning("DF is less than 5 lines")

            df['score'] = 1000 * (1-model.predict_proba(df)[:,1])

            output = data_path+'/output.csv'

            df.to_csv(output, index=False)

            return output
    """

    def __init__(self, model_type: str) -> None:
        if model_type.lower() not in ["sync", "async"]:
            raise InputError(
                f"Invalid model_type {model_type}. Valid options are Sync or Async"
            )

        self.model_type = model_type
        self.__levels = ["OUTPUT", "DEBUG", "WARNING", "ERROR"]
        self.__data = ""

    def __log(self, level: str, message: str):
        """
        Logger base method used by others.

        Parameters
        -----------
        level: str
            Log level (must be one used when initiating the logger)
        message: str
            Message that will be logged
        """

        if level in self.__levels:
            log_message = f"[{level}]{message}[{level}]"

            if self.model_type.lower() == "sync":
                self.__data += log_message

            else:
                base_path = os.getenv("BASE_PATH")
                exec_id = os.getenv("EXECUTION_ID")
                if base_path and "host" not in exec_id:
                    with open(
                        f"{base_path}/{exec_id}/output/execution.log", "a"
                    ) as file:
                        file.write(log_message + "\n")
                print(log_message)

        else:
            raise InputError(
                f"Invalid level {level}. Valid options are {' '.join(self.__levels)}"
            )

    def debug(self, message: str) -> None:
        """
        Logs a DEBUG message.

        Parameters
        ----------
        message: str
            Message that will be logged
        """
        self.__log("DEBUG", message)

    def warning(self, message: str) -> None:
        """
        Logs a WARNING message.

        Parameters
        ----------
        message: str
            Message that will be logged
        """
        self.__log("WARNING", message)

    def error(self, message: str) -> None:
        """
        Logs a ERROR message.

        Parameters
        ----------
        message: str
            Message that will be logged
        """
        self.__log("ERROR", message)

    def callback(self, output: Union[str, int, float, list, dict]) -> str:
        """
        Compile the logs with the response for Sync models only. Should be the return of function being executed.
        This output should be able to be parsed as a JSON, so if you are using a non-primitive type as your return, make sure it can be parsed by `json.dumps`.

        Example
        -------
        .. code-block:: python

            def score(data, base_path):
                logger = Logger('Sync')

                logger.debug("USING LOGGER")

                model = load(base_path+"/model.pkl")

                df = pd.DataFrame(data=json.loads(data), index=[0])

                return logger.callback({"score": 1000 * (1-float(model.predict_proba(df)[0,1]))})

        Parameters
        ----------
        output: str
            Output of the function being executed.
        """

        if self.model_type.lower() == "sync":
            if isinstance(output, (dict, list)):
                output = "[OUTPUT]" + json.dumps(output) + "[OUTPUT]"
            elif isinstance(output, (int, float)):
                output = "[OUTPUT]" + str(output) + "[OUTPUT]"
            else:
                raise InputError("Invalid type for logger callback")

            self.__log("OUTPUT", output)
            return self.__data
        else:
            raise InputError("callback function should only used in Sync models")
