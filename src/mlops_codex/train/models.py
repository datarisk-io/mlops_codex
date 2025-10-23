from pydantic import BaseModel, Field


class MLOpsExperiment(BaseModel):
    """ """

    training_hash: str
    experiment_name: str
    model_type: str
    group: str


class MLOpsTrainExecution(BaseModel):
    """ """

    experiment: MLOpsExperiment = Field(..., alias='experiment')
    execution_id: int = Field(alias='execution-id', gt=0)
