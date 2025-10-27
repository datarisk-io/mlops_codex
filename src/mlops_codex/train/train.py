from pydantic import BaseModel

from mlops_codex.administrator.auth import AuthManager
from mlops_codex.exceptions.module_exceptions import TrainExecutionException
from mlops_codex.train import TrainExecution
from mlops_codex.train.assemblers import (
    assemble_automl_request_content,
    assemble_custom_request_content,
    assemble_external_training_request_content,
)
from mlops_codex.train.client import execute, register, status, upload
from mlops_codex.train.models import MLOpsExperiment, MLOpsTrainExecution
from mlops_codex.train.validators import is_valid_model_type
from mlops_codex.utils.helpers import wait
from mlops_codex.utils.services_status import ExecutionStatus

assemblers = {
    'Custom': assemble_custom_request_content,
    'AutoML': assemble_automl_request_content,
    'External': assemble_external_training_request_content,
}


class MLOpsTrainClient(BaseModel):
    """
    Train class to connect with the Train module in Datarisk MLOps API.

    Args:
        auth (AuthManager): Auth manager.
    """

    auth: AuthManager

    def setup_project_experiment(
        self, experiment_name: str, model_type: str, group: str
    ) -> MLOpsExperiment:
        """
        Set up a new project experiment.

        Args:
            experiment_name (str): Name of the new experiment.
            model_type (str): Type of the model.
            group (str): Name of the group.
        Returns:
            (MLOpsExperiment): New experiment.
        """
        is_valid_model_type(model_type)

        data = {'experiment_name': experiment_name, 'model_type': model_type}
        training_hash = register(group=group, data=data, headers=self.auth.header)

        return MLOpsExperiment(
            training_hash=training_hash,
            experiment_name=experiment_name,
            group=group,
            model_type=model_type,
        )

    def run(
        self,
        experiment: MLOpsExperiment,
        train_type: TrainExecution,
        wait_ready: bool = False,
    ) -> MLOpsTrainExecution:
        """
        Run the training type in a given experiment.

        Args:
            experiment (MLOpsExperiment): Experiment to run.
            train_type (TrainExecution): Train execution to run.
            wait_ready (bool, optional): Whether to wait for the training to be ready.

        Returns:
            (MLOpsTrainExecution): Train execution.
        """
        assembler = assemblers[repr(train_type)]
        data, files = assembler(**train_type.model_dump())

        execution_id = upload(
            group=experiment.group,
            training_hash=experiment.training_hash,
            data=data,
            files=files,
            headers=self.auth.header,
        )

        execute(
            group=experiment.group,
            training_hash=experiment.training_hash,
            execution_id=execution_id,
            headers=self.auth.header,
        )

        if wait_ready:
            json_response = wait(
                f=status,
                valid_status=[ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED],
                status_enum=ExecutionStatus,
                status_key='Status',
                group=experiment.group,
                execution_id=execution_id,
                headers=self.auth.header,
            )

            execution_status = ExecutionStatus(json_response['Status'])

            if execution_status == ExecutionStatus.FAILED:
                raise TrainExecutionException()

        return MLOpsTrainExecution(
            experiment=experiment,
            execution_id=execution_id,
            auth=self.auth,
        )

    def deploy(self, *args, **kwargs): ...
