from typing import Union

from .models import CustomTrain, MLOpsExperiment, MLOpsTrainExecution

TrainExecution = Union[MLOpsTrainExecution, CustomTrain]

__all__ = ['MLOpsExperiment', 'MLOpsTrainExecution', 'CustomTrain', 'TrainExecution']
