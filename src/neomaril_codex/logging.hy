(import datetime)
(import os)

(defclass Logger [object]
  "Neomaril custom logger for model scripts.
  Neomaril has a log parser that clean logs and tries to find useful information.
  Since there are a lot of logger types is hard to create the perfect parser, so using this one helps our model script being parsed by Neomaril and sending cleaner messages"
  
  (defn __init__ [self [levels ["INFO" "DEBUG" "WARNING" "ERROR"]] [file False] [filePath "./"]]
    "Neomaril custom logger for model scripts.
     Neomaril has a log parser that clean logs and tries to find useful information.
     Since there are a lot of logger types is hard to create the perfect parser, so using this one helps our model script being parsed by Neomaril and sending cleaner messages

    Args:
        levels (list, optional): Log levels allowed. You can create new levels, and the logger will make sure the right format is being used. Defaults to ['INFO', 'DEBUG', 'WARNING', 'ERROR'].
        file (bool, optional): Should the logs be saved in a file. Defaults to False.
        filePath (str, optional): Path where the log file will be save. Defaults to './'."

    (setv self.levels (list (map (fn [x] (hy.models.Symbol x)) levels)))
    (setv self.logToFile file)
    (setv self.filePath filePath))
  
  (defn __level? [self symbol]
    (if (in (hy.models.Symbol symbol) self.levels)
        True
        False))
  
  (defn log [self level message]
    "Base logger function used by others. Use for custom levels.

    Args:
        level (str): Log level (must be one used when initiating the logger)
        message (str): Message that will be logged"
    
    (if (self.__level? level)
        (do 
          (if self.logToFile
            (do
              (os.makedirs self.filePath :exist_ok True) 
              (with [file (open (+ self.filePath (str (datetime.date.today)) ".log") "a")]
                  (.write file (+ "[" level "] - " (str (datetime.datetime.now)) " - " message " [" level "]" "\n"))
                  (.close file))))
          (print :flush True (+ "[" level "] - " (str (datetime.datetime.now)) " - " message " [" level "]")))
        (raise (Exception (+ "Invalid level on class definition '" level "'")))))

  (defn info [self message]
    "Short for .log('INFO', message)

    Args:
        message (str): Message that will be logged"
    (self.log 'INFO message))

  (defn debug [self message]
    "Short for .log('DEBUG', message)

    Args:
        message (str): Message that will be logged"
    (self.log 'DEBUG message))

  (defn warning [self message]
    "Short for .log('WARNING', message)

    Args:
        message (str): Message that will be logged"
    (self.log 'WARNING message))

  (defn error [self message]
    "Short for .log('ERROR', message)

    Args:
        message (str): Message that will be logged"
    (self.log 'ERROR message)))