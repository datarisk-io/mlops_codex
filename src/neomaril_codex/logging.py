import time
import os

class Logger(object):
  """Neomaril custom logger for model scripts.
  Neomaril has a log parser that clean logs and tries to find useful information.
  Since there are a lot of logger types is hard to create the perfect parser, so using this one helps our model script being parsed by Neomaril and sending cleaner messages"""
  
  def __init__(self, levels=[], file=False, filePath="./"):
    """Neomaril custom logger for model scripts.
     Neomaril has a log parser that clean logs and tries to find useful information.
     Since there are a lot of logger types is hard to create the perfect parser, so using this one helps our model script being parsed by Neomaril and sending cleaner messages

    Args:
        levels (list, optional): Log levels allowed. You can create new levels, and the logger will make sure the right format is being used. Defaults to ['INFO', 'DEBUG', 'WARNING', 'ERROR'].
        file (bool, optional): Should the logs be saved in a file. Defaults to False.
        filePath (str, optional): Path where the log file will be save. Defaults to './'."""

    self.levels=levels+["OUTPUT", "INFO", "DEBUG", "WARNING", "ERROR"]
    self.logToFile=file
    self.filePath=filePath
  
  def __level(self, symbol):
    return symbol in self.levels

  
  def log(self, level, message):
    """Base logger function used by others. Use for custom levels.

    Args:
        level (str): Log level (must be one used when initiating the logger)
        message (str): Message that will be logged"""
    
    if self.__level(level):
      log_message = f"[{level}]{message}[{level}]"
      if self.logToFile:
        os.makedirs(self.filePath, exist_ok=True)
        
        with open(self.filePath+str(time.time()).replace(".", "")+".log", "a") as file:
            file.write(log_message+"\n")
        
      print(log_message)
    else:
      raise Exception(f"Invalid level on class definition '{level}'")

  def info(self, message):
    """Short for .log('INFO', message)

    Args:
        message (str): Message that will be logged"""
    self.log('INFO', message)

  def debug(self, message):
    """Short for .log('DEBUG', message)

    Args:
        message (str): Message that will be logged"""
    self.log('DEBUG', message)

  def warning(self, message):
    """Short for .log('WARNING', message)

    Args:
        message (str): Message that will be logged"""
    self.log('WARNING', message)

  def error(self, message):
    """Short for .log('ERROR', message)

    Args:
        message (str): Message that will be logged"""
    self.log('ERROR', message)