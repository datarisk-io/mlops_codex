from typing import Union

from .models import AutoMLTrain, CustomTrain, External, MLOpsExperiment, MLOpsTrainExecution

TrainExecution = Union[AutoMLTrain, CustomTrain, External, MLOpsTrainExecution]

__all__ = [
    'MLOpsExperiment',
    'MLOpsTrainExecution',
    'CustomTrain',
    'TrainExecution',
    'AutoMLTrain',
    'External',
]
