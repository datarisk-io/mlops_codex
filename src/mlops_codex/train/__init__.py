from typing import Union

from .models import MLOpsExperiment, MLOpsTrainExecution, CustomTrain

TrainExecution = Union[MLOpsTrainExecution, CustomTrain]

__all__ = ['MLOpsExperiment', 'MLOpsTrainExecution', 'CustomTrain']
