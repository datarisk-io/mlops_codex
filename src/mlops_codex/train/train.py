from pydantic import BaseModel, FilePath

from mlops_codex.administrator.auth import AuthManager
from mlops_codex.exceptions.module_exceptions import TrainExecutionException
from mlops_codex.train import TrainExecution
from mlops_codex.train.assemblers import (
    assemble_automl_request_content,
    assemble_custom_request_content,
    assemble_external_training_request_content,
)
from mlops_codex.train.client import execute, register, status, upload, promote, search
from mlops_codex.train.models import MLOpsExperiment, MLOpsTrainExecution
from mlops_codex.train.validators import is_valid_model_type
from mlops_codex.utils.helpers import wait
from mlops_codex.utils.logger_configuration import get_logger
from mlops_codex.utils.services_status import ExecutionStatus
from mlops_codex.utils.validations import str_to_path


logger = get_logger()

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

    def __get_repeated_thash(
        self, model_type: str, experiment_name: str, group: str
    ) -> str | None:
        """Look for a previous train experiment.
        Args:
            experiment_name (str): name given to the training
            model_type (str): type of the model being trained
            group (str): name of the group, previous created, where the training will be inserted

        Returns:
            (str | None): Training hash if found, otherwise None
        """
        response = search(self.auth.header)

        results = response.get('Results')
        for result in results:
            condition = (
                result['ExperimentName'] == experiment_name
                and result['GroupName'] == group
                and result['ModelType'] == model_type
            )
            if condition:
                logger.info('Found experiment with same attributes...')
                return result['TrainingHash']
        return None

    def setup_project_experiment(
        self, experiment_name: str, model_type: str, group: str, force: bool = False
    ) -> MLOpsExperiment:
        """
        Set up a new project experiment.

        Args:
            experiment_name (str): Name of the new experiment.
            model_type (str): Type of the model.
            group (str): Name of the group.
            force (bool, optional): Whether to recreate an existing experiment.
        Returns:
            (MLOpsExperiment): New experiment.
        """
        is_valid_model_type(model_type)

        training_hash = self.__get_repeated_thash(
            model_type=model_type, experiment_name=experiment_name, group=group
        )

        if force or training_hash is None:
            msg = (
                'The experiment you are creating has identical name, group, and model type attributes to an existing one. '
                'Since forced creation is active, we will continue with the process as specified'
                if force
                else 'Could not find experiment. Creating a new one...'
            )
            logger.info(msg)
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

    def deploy(
        self,
        execution: MLOpsTrainExecution,
        source: FilePath | str,
        schema: FilePath | str,
        name: str,
        model_reference: str,
        operation: str,
        input_type: str,
    ) -> str:
        """
        Deploy a trained model.

        Args:
            execution (MLOpsTrainExecution): Train execution to deploy
            source (FilePath | str): Path to the .py script with an entry point function
            schema (FilePath | str): Path to a json file with a sample of the input for the entry point function
            name (str): Name of the model, should be not null, case-sensitive, have between 3 and 32 characters, that could be alphanumeric including accentuation (for example: 'é', à', 'ç','ñ') and space
            model_reference (str): Name of the entry point function at the source file
            operation (str): Defines how the model will be treated at the API. It can be:
                             - Sync: for synchronous models that will work as real-time executions;
                             - Async: for asynchronous models that will not deliver real-time executions.
            input_type (str): The type of the input that the model expect

        Returns:
            (str): Model hash
        """

        data = {
            'name': name,
            'operation': operation,
            'input_type': input_type,
            'model_reference': model_reference,
        }

        source = str_to_path(source)
        schema = str_to_path(schema)

        files = [
            ('source', (source.name, open(source, 'rb'))),
            ('schema', (schema.name, open(schema, 'rb'))),
        ]

        model_hash = promote(
            group=execution.experiment.group,
            training_hash=execution.training_hash,
            execution_id=execution.execution_id,
            data=data,
            files=files,
            headers=self.auth.header,
        )
        return model_hash
