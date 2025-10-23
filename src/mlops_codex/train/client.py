from http import HTTPStatus

from pydantic import BaseModel

from mlops_codex.base.client import send_http_request
from mlops_codex.train.assemblers import (
    assemble_custom_request_content,
    assemble_automl_request_content,
)
from mlops_codex.train.models import MLOpsExperiment
from mlops_codex.train.validators import is_valid_model_type
from mlops_codex.utils.helpers import wait
from mlops_codex.utils.services_status import ExecutionStatus
from mlops_codex.utils.urls import TrainingUrl


assemblers = {
    'Custom': assemble_custom_request_content,
    'AutoML': assemble_automl_request_content,
}


def register(data: dict[str, str], group: str, **kwargs) -> str:
    response = send_http_request(
        url=TrainingUrl.REGISTER_URL.format(group_name=group),
        method='POST',
        successful_code=HTTPStatus.CREATED,
        data=data,
        **kwargs,
    ).json()

    print(response['Message'])

    return response['TrainingHash']


def upload(group: str, training_hash: str, **kwargs) -> int:
    response = send_http_request(
        url=TrainingUrl.UPLOAD_URL.format(
            group_name=group, training_hash=training_hash
        ),
        method='POST',
        successful_code=HTTPStatus.CREATED,
        **kwargs,
    ).json()

    print(response['Message'])

    return response['ExecutionId']


def execute(group: str, training_hash: str, execution_id: int, **kwargs) -> None:
    response = send_http_request(
        url=TrainingUrl.EXECUTE_URL.format(
            group_name=group, training_hash=training_hash, execution_id=execution_id
        ),
        method='GET',
        successful_code=HTTPStatus.OK,
        **kwargs,
    ).json()

    print(response['Message'])


def status(group: str, execution_id: int, **kwargs):
    response = send_http_request(
        url=TrainingUrl.STATUS_URL.format(group_name=group, execution_id=execution_id),
        method='GET',
        successful_code=HTTPStatus.OK,
        **kwargs,
    ).json()

    str_status = response['Status']
    print(response['Message'])
    return ExecutionStatus(str_status)


class MLOpsTrainClient(BaseModel):
    """
    Train class to connect with the Train module in Datarisk MLOps API.
    """

    @staticmethod
    def setup_project_experiment(
        experiment_name: str, model_type: str, group: str, **kwargs
    ) -> MLOpsExperiment:
        """
        Set up a new project experiment.

        Args:
            experiment_name (str): Name of the new experiment.
            model_type (str): Type of the model.
            group (str): Name of the group.
            **kwargs:
                Additional arguments to send an HTTP request.
        Returns:

        """
        is_valid_model_type(model_type)

        data = {'experiment_name': experiment_name, 'model_type': model_type}
        training_hash = register(group=group, data=data, **kwargs)

        return MLOpsExperiment(
            training_hash=training_hash,
            experiment_name=experiment_name,
            group=group,
            model_type=model_type,
        )

    # @staticmethod
    # def run(experiment: MLOpsExperiment, wait_ready: bool = False, **kwargs):
    #     training_type = kwargs.pop('training_type')
    #     data, files = assemblers[training_type](**kwargs)
    #
    #     execution_id = upload(
    #         group=experiment.group,
    #         training_hash=experiment.training_hash,
    #         data=data,
    #         files=files,
    #         login_token=login_token,
    #     )
    #
    #     execute(
    #         group=experiment.group,
    #         training_hash=experiment.training_hash,
    #         execution_id=execution_id,
    #         login_token=login_token,
    #     )
    #
    #     if wait_ready:
    #         wait(
    #             f=status,
    #             valid_status=[ExecutionStatus.SUCCESS, ExecutionStatus.FAILED],
    #             status_enum=ExecutionStatus,
    #             login_token=self.login_token,
    #             group=group,
    #             execution_id=execution_id,
    #         )
    #
    # def deploy(self, *args, **kwargs): ...
