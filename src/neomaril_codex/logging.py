import time
import os

from neomaril_codex.exceptions import InputError

class Logger(object):
    """Neomaril custom logger for model scripts.
    Neomaril has a log parser that clean logs and tries to find useful information.
    Since there are a lot of logger types is hard to create the perfect parser, so using this one helps your model script being parsed by Neomaril and sending cleaner messages"""
    
    def __init__(self, model_type):
        """Neomaril custom logger for model scripts.
        Neomaril has a log parser that clean logs and tries to find useful information.
        Since there are a lot of logger types is hard to create the perfect parser, so using this one helps your model script being parsed by Neomaril and sending cleaner messages

        Args:
        model_type (str): Model operation type. Could be Sync or Async
        """
        self.operation = model_type
        self.levels=["OUTPUT", "DEBUG", "WARNING", "ERROR"]
        self.data = ''
        
    def __log(self, level, message):
        """Base logger function used by others.

        Args:
            level (str): Log level (must be one used when initiating the logger)
            message (str): Message that will be logged"""
        
        if level in self.levels:
            log_message = f"[{level}]{message}[{level}]"

            if self.operation.title() == 'Sync':
                self.data += log_message

            else: 
                base_path = os.getenv('BASE_PATH')
                exec_id = os.getenv('EXECUTION_ID')
                if base_path and exec_id:
                    with open(f"{base_path}/{exec_id}/output/execution.log", "a") as file:
                        file.write(log_message+"\n")
                print(log_message)

        else:
            raise InputError(f'Invalid level {level}. Valid options are {" ".join(self.levels)}')


    def debug(self, message):
        """Logs a DEBUG message

        Args:
                message (str): Message that will be logged"""
        self.__log('DEBUG', message)

    def warning(self, message):
        """Logs a WARNING message

        Args:
                message (str): Message that will be logged"""
        self.__log('WARNING', message)

    def error(self, message):
        """Logs a ERROR message

        Args:
                message (str): Message that will be logged"""
        self.__log('ERROR', message)


    def callback(self, output):
        """Logs a ERROR message

        Args:
                message (str): Message that will be logged"""
        if self.operation == "Sync":
            self.__log('OUTPUT', output)
            return self.data
        else:
            raise InputError('callback function should only used in Sync models')