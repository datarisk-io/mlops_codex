import json
import os
from datetime import datetime
from typing import Optional, Tuple

import yaml
from dotenv import find_dotenv, load_dotenv

from mlops_codex.base import BaseMLOps
from mlops_codex.exceptions import ModelError, PipelineError, TrainingError
from mlops_codex.logger_config import get_logger
from mlops_codex.model import MLOpsModel, MLOpsModelClient
from mlops_codex.training import MLOpsTrainingClient

logger = get_logger()


class MLOpsPipeline(BaseMLOps):
    """
    Class to construct and orchestrates the flow data of the models inside MLOps.

    Parameters
    ----------
    login: str
        Login for authenticating with the client. You can also use the env variable MLOPS_USER to set this
    password: str
        Password for authenticating with the client. You can also use the env variable MLOPS_PASSWORD to set this
    group: str
        Group the model is inserted
    url: str
        URL to MLOps Server. Default value is https://neomaril.datarisk.net/, use it to test your deployment first before changing to production. You can also use the env variable MLOPS_URL to set this
    python_version: str, default="3.9"
        Python version for the model environment. Available versions are 3.8, 3.9, 3.10.

    Example
    --------

    .. code-block:: python

        from mlops_codex.pipeline import MLOpsPipeline

        pipeline = MLOpsPipeline.from_config_file('./samples/pipeline.yml')
        pipeline.register_monitoring_config(directory = "./samples/monitoring", preprocess = "preprocess.py", preprocess_function = "score", shap_function = "score", config = "configuration.json", packages = "requirements.txt")
        pipeline.start()
        pipeline.run_monitoring('2', 'Mb29d61da4324a39a8bc2e0946f213b4959643916d354bf39940de2124f1e9d8')
    """

    def __init__(
        self,
        *,
        group: str,
        login: Optional[str] = None,
        password: Optional[str] = None,
        url: Optional[str] = None,
        python_version: str = "3.10",
    ) -> None:
        super().__init__(login=login, password=password, url=url)

        self.__start = False
        self.group = group
        self.python_version = python_version
        self.train_config = None
        self.deploy_config = None
        self.monitoring_config = None
        self.__training = None
        self.__training_run = None
        self.__training_id = None
        self.__model = None
        self.__model_id = None

    @staticmethod
    def __try_create_group(client, group: str):
        groups = client.list_groups()

        if group not in [g["Name"] for g in groups]:
            client.create_group(name=group, description="Created with Codex Pipeline")

    def register_train_config(self, **kwargs) -> None:
        """
        Set the files for configure the training

        Parameters
        ----------
        kwargs: list or dict
            List or dictionary with the necessary files for training
        """
        self.train_config = kwargs

    def register_deploy_config(self, **kwargs) -> None:
        """
        Set the files for configure the deployment

        Parameters
        ----------
        kwargs: list or dict
            List or dictionary with the necessary files for deploy
        """
        self.deploy_config = kwargs

    def register_monitoring_config(self, **kwargs) -> None:
        """
        Set the files for configure the monitoring

        Parameters
        ----------
        kwargs: list or dict
            List or dictionary with the necessary files for monitoring
        """
        self.monitoring_config = kwargs

    @staticmethod
    def from_config_file(path):
        """
        Load the configuration files for orchestrate the model

        Parameters
        ----------
        path: str
            Path of the configuration file, but it could be a dict

        Raises
        ------
        PipelineError
            Undefined credentials

        Returns
        -------
        MLOpsPipeline
            The new pipeline
        """
        with open(path, "rb") as stream:
            try:
                conf = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        loaded = load_dotenv()
        # Something's when running as a script the default version might not work
        if not loaded:
            load_dotenv(find_dotenv(usecwd=True))
        logger.info("Loading .env")

        login = os.getenv("MLOPS_USER")
        if not login:
            raise PipelineError(
                "When using a config file the environment variable MLOPS_USER must be defined"
            )

        password = os.getenv("MLOPS_PASSWORD")
        if not password:
            raise PipelineError(
                "When using a config file the environment variable MLOPS_PASSWORD must be defined"
            )

        url = os.getenv("MLOPS_URL", conf.get("url"))

        pipeline = MLOpsPipeline(
            group=conf["group"],
            login=login,
            password=password,
            url=url,
            python_version=conf["python_version"],
        )

        if "training" in conf.keys():
            pipeline.register_train_config(**conf["training"])

        if "deploy" in conf.keys():
            pipeline.register_deploy_config(**conf["deploy"])

        if "monitoring" in conf.keys():
            pipeline.register_monitoring_config(**conf["monitoring"])

        return pipeline

    def run_training(self) -> Tuple[str, str]:
        """
        Run the training process

        Raises
        ------
        TrainingError
            Training has failed

        Returns
        -------
        tuple[str, str]
            A tuple with the 'training_id' and the 'exec_id'

        Example
        -------
        >>> pipeline.run_training(run_name=,training_type=)
        """
        logger.info("Running training")
        client = MLOpsTrainingClient(
            login=self.credentials[0], password=self.credentials[1], url=self.base_url
        )
        self.__try_create_group(client, self.group)

        conf = self.train_config

        self.__training = client.create_training_experiment(
            experiment_name=conf["experiment_name"],
            model_type=conf["model_type"],
            group=self.group,
        )

        PATH = conf["directory"]
        run_name = conf.get("run_name", "Pipeline run " + str(datetime.now()))
        extra_files = conf.get("extra")

        if conf["training_type"] == "Custom":
            self.__training_run = self.__training.run_training(run_name=run_name, training_type=conf["training_type"],
                                                               train_data=os.path.join(PATH, conf["data"]),
                                                               requirements_file=os.path.join(PATH, conf["packages"]),
                                                               source_file=os.path.join(PATH, conf["source"]),
                                                               python_version=str(self.python_version),
                                                               training_reference=conf["train_function"], extra_files=(
                    [os.path.join(PATH, e) for e in extra_files]
                    if extra_files
                    else None
                ), wait_complete=True)

        elif conf["training_type"] == "AutoML":
            self.__training_run = self.__training.run_training(run_name=run_name,
                                                               training_type=conf["training_type"],
                                                               train_data=os.path.join(PATH, conf["data"]),
                                                               conf_dict=os.path.join(PATH, conf["config"]),
                                                               wait_complete=True)

        elif conf["training_type"] == "External":
            self.__training_run = self.__training.run_training(run_name=run_name, training_type=conf["training_type"],
                                                               python_version=conf["python_version"],
                                                               model_file=conf["model_file"], extra_files=(
                    [os.path.join(PATH, e) for e in extra_files]
                    if extra_files
                    else None
                ), wait_complete=True)
        else:
            raise TrainingError(
                f"Invalid training_type {conf['training_type']}. Valid options are Custom, AutoML and External"
            )

        status = self.__training_run.status
        if status == "Succeeded":
            logger.info("Model training finished")
            return self.__training.training_id, self.__training_run.exec_id
        else:
            raise TrainingError("Training failed")

    def run_deploy(self, training_id: Optional[str] = None) -> str:
        """
        Run the deployment process

        Parameters
        ----------
        training_id: Optional[str], optional
            The id for the training process that you want to deploy now

        Raises
        ------
        ModelError
            Deploy has failed

        Returns
        -------
        str
            The new Model id (hash)

        Example
        -------
        >>> pipeline.run_deploy('Mb29d61da4324a39a8bc2e0946f213b4959643916d354bf39940de2124f1e9d8')
        """
        conf = self.deploy_config
        PATH = conf["directory"]
        extra_files = conf.get("extra")

        if training_id:
            logger.info("Deploying scorer from training")

            model_name = conf.get(
                "name", self.__training_run.execution_data.get("ExperimentName", "")
            )

            if self.__training_run.execution_data["TrainingType"] == "Custom":
                self.__model = self.__training_run.promote_model(
                    model_name=model_name,
                    model_reference=conf["score_function"],
                    source_file=os.path.join(PATH, conf["source"]),
                    input_type=conf["input_type"],
                    extra_files=(
                        [os.path.join(PATH, e) for e in extra_files]
                        if extra_files
                        else None
                    ),
                    env=os.path.join(PATH, conf["env"]) if conf.get("env") else None,
                    schema=os.path.join(PATH, conf["schema"]),
                    operation=conf["operation"],
                )

            elif self.__training_run.model_type == "AutoML":
                self.__model = self.__training_run.promote_model(
                    model_name=model_name, operation=conf["operation"]
                )

        else:
            logger.info("Deploying scorer")
            client = MLOpsModelClient(
                login=self.credentials[0],
                password=self.credentials[1],
                url=self.base_url,
            )
            self.__try_create_group(client, self.group)

            self.__model = client.create_model(
                model_name=conf.get("name"),
                model_reference=conf["score_function"],
                source_file=os.path.join(PATH, conf["source"]),
                model_file=os.path.join(PATH, conf["model"]),
                requirements_file=os.path.join(PATH, conf["packages"]),
                extra_files=(
                    [os.path.join(PATH, e) for e in extra_files]
                    if extra_files
                    else None
                ),
                env=os.path.join(PATH, conf["env"]) if conf.get("env") else None,
                schema=(
                    os.path.join(PATH, conf["schema"]) if conf.get("schema") else None
                ),
                operation=conf["operation"],
                input_type=conf["input_type"],
                group=self.group,
            )

        while self.__model.status() == "Building":
            self.__model.wait_ready()

        if self.__model.status() == "Deployed":
            logger.info("Model deployement finished")
            return self.__model.model_hash

        else:
            raise ModelError(
                "Model deployement failed: " + self.__model.get_logs(routine="Host")[0]
            )

    def run_monitoring(
        self, *, training_exec_id: Optional[str] = None, model_id: Optional[str] = None
    ):
        """
        Run the monitoring process

        Parameters
        ----------
        training_exec_id: Optional[str], optional
            The id for the training execution process that you want to monitore now
        model_id: Optional[str], optional
            Model hash

        Example
        -------
        >>> pipeline.run_monitoring('2', 'Mb29d61da4324a39a8bc2e0946f213b4959643916d354bf39940de2124f1e9d8')
        """
        logger.info("Configuring monitoring")

        conf = self.monitoring_config
        PATH = conf["directory"]

        if training_exec_id:
            with open(os.path.join(PATH, conf["config"]), "r+") as f:
                conf_dict = json.load(f)
                f.seek(0)
                conf_dict["TrainData"]["MLOpsTrainingExecution"] = training_exec_id
                json.dump(conf_dict, f)
                f.truncate()

        model = MLOpsModel(
            name=conf.get("name"),
            model_hash=conf.get("model_hash", model_id),
            group=self.group,
            login=self.credentials[0],
            password=self.credentials[1],
            group_token=os.getenv("MLOPS_GROUP_TOKEN"),
            url=self.base_url,
        )

        model.register_monitoring(
            preprocess_reference=conf["preprocess_function"],
            shap_reference=conf["shap_function"],
            configuration_file=os.path.join(PATH, conf["config"]),
            preprocess_file=os.path.join(PATH, conf["preprocess"]),
            requirements_file=(
                os.path.join(PATH, conf["packages"]) if conf.get("packages") else None
            ),
        )

    def start(self):
        """
        Start the pipeline for the model orchestration

        Raises
        ------
        PipelineError
            Cannot start pipeline without configuration

        Example
        -------
        >>> pipeline = MLOpsPipeline.from_config_file('./samples/pipeline.yml').start()
        """
        if (
            (not self.train_config)
            and (not self.deploy_config)
            and (not self.monitoring_config)
        ):
            raise PipelineError("Cannot start pipeline without configuration")

        if self.train_config:
            self.__training_id = self.run_training()

        if self.deploy_config:
            self.__model_id = self.run_deploy(training_id=self.__training_id)

        if self.monitoring_config:
            self.run_monitoring(
                training_exec_id=(
                    self.__training_id[1] if self.__training_id else None
                ),
                model_id=self.__model_id,
            )
        self.__start = True

    @property
    def training(self):
        if not self.__start:
            raise PipelineError(
                "Pipeline didnt run. Run it first before trying to access the training"
            )

        if not self.train_config:
            raise PipelineError("Training configuration not found.")

        if self.__training:
            return self.__training

    @property
    def training_run(self):
        if not self.__start:
            raise PipelineError(
                "Pipeline didnt run. Run it first before trying to access the training run"
            )

        if not self.train_config:
            raise PipelineError("Training configuration not found.")

        if self.__training_run:
            return self.__training_run

    @property
    def model(self):
        if not self.__start:
            raise PipelineError(
                "Pipeline didnt run. Run it first before trying to access the model"
            )

        if not self.deploy_config:
            raise PipelineError("Model deployment configuration not found.")

        if self.__model:
            return self.__model
