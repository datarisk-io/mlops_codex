import hy
from neomaril_logger.loggerBuilder import Logger

def create (file = False):
    return Logger(file=file)
