from typing import Union

from .models import CustomTrain, External, MLOpsExperiment, MLOpsTrainExecution

TrainExecution = Union[MLOpsTrainExecution, CustomTrain, External]

__all__ = ['MLOpsExperiment', 'MLOpsTrainExecution', 'CustomTrain', 'TrainExecution']
