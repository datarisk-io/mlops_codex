(import datetime)

(defclass Logger [object]
  (defn __init__ [self [levels ["DEBUG" "WARNING" "ERROR"]] [file False]]
    (setv self.levels (list (map (fn [x] (hy.models.Symbol x)) levels)))
    (setv self.logToFile file))
  
  (defn level? [self symbol]
    (if (in (hy.models.Symbol symbol) self.levels)
        True
        False))
  
  (defn log [self level message]
    
    (if (self.level? level)
        (if self.logToFile
            (with [file (open (+ (str (datetime.datetime.now)) ".log") "a")]
              (.write file (+ "[" level "]: " message "\n"))
              (.close file))
            (print :flush True (+ "[" level "]: " message)))
        (raise (Exception (+ "Invalid level on class definition '" level "'")))))

  (defn logDebug [self message]
    (self.log 'DEBUG message))

  (defn logWarning [self message]
    (self.log 'WARNING message))

  (defn logError [self message]
    (self.log 'ERROR message)))
