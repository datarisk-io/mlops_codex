#!/usr/bin/env python
# coding: utf-8
import json
import os
import re
import sys
from contextlib import contextmanager
from typing import Any, Optional, Union

import cloudpickle
import numpy as np
import pandas as pd
from lazy_imports import try_import

from mlops_codex.__utils import parse_json_to_yaml
from mlops_codex.base import BaseMLOps, BaseMLOpsClient
from mlops_codex.dataset import validate_dataset
from mlops_codex.datasources import MLOpsDataset
from mlops_codex.exceptions import (
    InputError,
    TrainingError,
)
from mlops_codex.http_request_handler import make_request, refresh_token
from mlops_codex.logger_config import get_logger
from mlops_codex.shared import constants
from mlops_codex.shared.utils import parse_data
from mlops_codex.training.base import ITrainingExecution
from mlops_codex.training.training_executions import (
    AutoMLTrainingExecution,
    CustomTrainingExecution,
    ExternalTrainingExecution,
)
from mlops_codex.validations import validate_group_existence

patt = re.compile(r"(\d+)")
logger = get_logger()


class MLOpsTrainingLogger:
    """A class for logging MLOps training runs.

    Example
    -------

    .. code-block:: python
        with training.log_train('Teste 1', X, y) as logger:
            pipe.fit(X, y)
            logger.save_model(pipe)

            params = pipe.get_params()
            params.pop('steps')
            params.pop('simpleimputer')
            params.pop('lgbmclassifier')
            logger.save_params(params)

            model_output = pd.DataFrame({"pred": pipe.predict(X), "proba": pipe.predict_proba(X)[:,1]})
            logger.save_model_output(model_output)

            auc = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
            f_score = cross_val_score(pipe, X, y, cv=5, scoring="f1")
            logger.save_metric(name='auc', value=auc.mean())
            logger.save_metric(name='f1_score', value=f_score.mean())

            logger.set_python_version('3.10')
    """

    def __init__(
        self,
        *,
        name: str,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        save_path: Optional[str] = None,
    ):
        self.name = name
        self.X_train = X_train
        self.y_train = y_train
        self.output = None
        self.model = None
        self.metrics = {}
        self.params = {}
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.extras = []

        # Paths
        self.features_file = None
        self.target_file = None
        self.output_file = None
        self.metrics_file = None
        self.params_file = None
        self.requirements = None
        self.model_file = None

        if not save_path:
            dir_name = self.name.replace(" ", "_")
            save_path = f"./{dir_name}"

        os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path

    def _processing_logging_inputs(self):
        """
        Processing of everything that be logged and return object
        """

        self.__set_params()
        self.params_file = self.__to_json("params", self.params)

        self.features_file = self.__to_parquet(
            output_filename="features",
            input_data=self.__parse_data_objects(self.X_train),
        )

        self.target_file = self.__to_parquet(
            output_filename="target", input_data=self.__parse_data_objects(self.y_train)
        )

        self.output_file = self.__to_parquet(
            output_filename="predictions",
            input_data=self.__parse_data_objects(self.output),
        )

        if self.model:
            self.model_file = self.__to_pickle(
                output_filename="model", input_data=self.model
            )

        if self.metrics:
            self.metrics_file = self.__to_json("metrics", self.metrics)

    def save_model(self, model):
        """
        Save the trained model to the logger.

        Parameters
        ----------
        model: object
            The trained model.
        """
        self.model = model

    def save_metric(self, *, name, value):
        """
        Save a metric to the logger.

        Parameters
        ----------
        name: str
            The name of the metric.
        value: float
            The value of the metric.
        """
        self.metrics[name] = value

    def save_model_output(self, output):
        """
        Save the model output to the logger.

        Parameters
        ----------
        output: object
            The output of the trained model.
        """
        self.output = output

    def set_python_version(self, version: str):
        """
        Set the Python version used to train the model.

        Parameters
        ----------
        version: str
            The Python version.
        """
        self.python_version = version

    def set_requirements(self, requirements: str):
        """
        Set the project requirements.

        Parameters
        ----------
        requirements: str
            The path of project requirements.
        """
        self.requirements = requirements

    def save_plot(
        self, *, fig: object, filename: str, dpi: int = 300, ext: str = "png"
    ):
        """
        Save plot graphic image to the logger.

        Parameters
        ----------
        fig: matplotlib.figure.Figure, seaborn.axisgrid.FacetGrid, seaborn.axes._subplots.AxesSubplot plotly.graph_objects.Figure.
            Figure object
        filename: str
            filename without extension (extension will be added automatically).
        dpi: int, default=300
            Resolution for matplotlib/seaborn plots. Default is 300.
        ext: str, default='png'
            File format to save (e.g., 'png', 'pdf', 'svg', 'html'). If None, defaults to 'png' for static images.

        Raises
        ------
        TypeError
            If the figure type is not supported.
        """

        filepath = f"{self.save_path}/{filename}.{ext}"

        with try_import() as _:
            import plotly

            if isinstance(fig, plotly.graph_objs.Figure):
                self.__save_plotly_plot(fig=fig, filepath=filepath)
                return

        with try_import() as _:
            import matplotlib.pyplot as plt
            import seaborn as sns

            if isinstance(fig, sns.axisgrid.FacetGrid) or isinstance(fig, plt.Figure):
                self.__save_seaborn_or_matplotlib_plot(
                    fig=fig, dpi=dpi, filepath=filepath
                )
                return

        raise TypeError("The plot only accepts plots of Matplotlib/Plotly/Seaborn")

    def __save_plotly_plot(self, fig, filepath):
        """
        Save a plotly figure to the logger.

        Parameters
        ----------
        fig: plotly.graph_objects.Figure
            The figure to save.
        filepath: str
            Path to the file to save.
        **kwargs:
            Extra keyword arguments passed to savefig() or write_image()/write_html().
        """

        if filepath.endswith("html"):
            fig.write_html(f"{filepath}")
            return

        fig.write_image(f"{filepath}")
        self.add_extra(extra=filepath)

    def __save_seaborn_or_matplotlib_plot(self, *, fig, dpi, filepath):
        fig.savefig(filepath, dpi=dpi)
        self.add_extra(extra=filepath)

    def set_extra(self, extra: list):
        """
        Set the extra files list.

        Parameters
        ----------
        extra: list
            A list of paths to the extra files.
        """
        self.extras = extra

    def add_extra(self, *, extra: Union[pd.DataFrame, str], filename: str = None):
        """
        Add an extra file in the extra file list.

        Parameters
        ----------
        extra: Union[pd.DataFrame, str]
            A path of an extra file or a list to include in extra file list.
        filename: Optional[str], optional
            A filename if the extra is a DataFrame.
        """

        if isinstance(extra, str):
            if os.path.exists(extra):
                self.extras.append(extra)
            else:
                raise FileNotFoundError("Extra file path not found!")
        elif isinstance(extra, pd.DataFrame):
            if filename:
                self.extras.append(
                    self.__to_parquet(output_filename=filename, input_data=extra)
                )
            else:
                raise InputError("Needs a filename to save the dataframe parquet.")
        else:
            raise InputError("Extra must be a Pandas DataFrame or a path.")

    def add_requirements(self, filename: str):
        """
        Add a requirement file.

        Parameters
        ----------
        filename: str
            The name of the output filename to save.
        """
        self.requirements = filename

    def __to_parquet(self, *, output_filename: str, input_data: pd.DataFrame):
        """
        Transform dataframe to parquet.

        Args:
            output_filename: The name of output filename to save.
            input_data: A pandas dataframe to save.
        """
        path = os.path.join(self.save_path, f"{output_filename}.parquet")
        input_data.to_parquet(path)
        self.add_extra(extra=path)
        return path

    def __to_json(self, output_filename: str, input_data: dict):
        """
        Transform dict to JSON.

        Args:
            output_filename: The name of the output filename to save.
            input_data: A dictionary to save.
        """
        path = os.path.join(self.save_path, f"{output_filename}.json")
        with open(path, mode="w", encoding="utf-8") as json_file:
            json.dump(input_data, json_file)
        self.add_extra(extra=path)
        return path

    def __to_pickle(self, *, output_filename: str, input_data):
        """
        Transform content to pickle.

        Args:
            output_filename: The name of the output filename to save.
            input_data: The content to save.
        """
        path = os.path.join(self.save_path, f"{output_filename}.pkl")
        with open(path, "wb") as f:
            cloudpickle.dump(input_data, f)
        self.add_extra(extra=path)
        return path

    def __set_params(self):
        """
        Set parameters for training.
        """
        missing = self.X_train.isna().sum()
        missing_dict = {
            k + "_missings": v
            for k, v in missing[missing > 0].describe().to_dict().items()
            if k != "count"
        }

        params = {
            "shape": self.X_train.shape,
            "cols_with_missing": len(missing[missing > 0]),
            "missing_distribution": missing_dict,
        }

        try:
            params["pipeline_steps"] = list(self.model.named_steps.keys())
        except Exception:
            params["pipeline_steps"] = [
                str(self.model.__class__).replace("<class '", "").replace("'>", "")
            ]

        if "get_all_params" in dir(self.model):
            hyperparameters = {
                f"hyperparam_{k}": str(v)
                for k, v in self.model.get_all_params().items()
                if k != "task_type"
            }
        elif "get_params" in dir(self.model):
            hyperparameters = {
                "hyperparam_" + k: str(v)
                for k, v in self.model.get_params().items()
                if k not in params["pipeline_steps"] + ["steps", "memory", "verbose"]
            }

            params = {**params, **hyperparameters}

        if len(self.y_train.value_counts()) < 10:
            target_proportion = self.y_train.value_counts() / len(self.y_train)
            target_proportion = target_proportion.to_dict()
            target_proportion = [
                {"target": k, "proportion": v} for k, v in target_proportion.items()
            ]
            params["target_proportion"] = target_proportion
        else:
            params["target_distribution"] = {
                k: v
                for k, v in self.y_train.describe().to_dict().items()
                if k != "count"
            }

        self.params = {**params, **self.params}

    @staticmethod
    def __parse_data_objects(obj: Any) -> pd.DataFrame:
        """
        Transform data types to dataframe
        """
        if isinstance(obj, pd.Series):
            return obj.to_frame()
        elif isinstance(obj, (np.ndarray, list, tuple, dict)):
            array_df = pd.DataFrame(obj)
            array_df.columns = [str(c) for c in array_df.columns]
            return array_df
        elif isinstance(obj, pd.DataFrame):
            return obj
        else:
            raise TypeError(f"{obj} couldn't be a DataFrame")


class MLOpsTrainingExperiment(BaseMLOps):
    """
    Class to manage models being trained inside MLOps

    Parameters
    ----------
    login: str
        Login for authenticating with the client. You can also use the env variable MLOPS_USER to set this
    password: str
        Password for authenticating with the client. You can also use the env variable MLOPS_PASSWORD to set this
    training_hash: str
        Training id (hash) from the experiment you want to access
    group: str
        Group the training is inserted.
    environment: str
        Flag that choose which environment of MLOps you are using. Test your deployment first before changing to production. Default is True
    executions: List[int]
        Ids for the executions in that training


    Raises
    ------
    TrainingError
        When the training can't be accessed in the server
    AuthenticationError
        Invalid credentials

    Example
    -------

    .. code-block:: python

        from mlops_codex.training import MLOpsTrainingClient
        from mlops_codex.base import MLOpsExecution

        client = MLOpsTrainingClient('123456')
        client.create_group('ex_group', 'Group for example purpose')
        training = client.create_training_experiment('Training example', 'Classification', 'ex_group')
        print(client.get_training(training.training_id, 'ex_group').training_data)

        data_path = './samples/train/'

        run = run = training.run_training('First test', data_path+'dados.csv', training_reference='train_model', training_type='Custom', python_version='3.9', requirements_file=data_path+'requirements.txt', wait_complete=True)

        print(run.get_training_execution(run.exec_id))
        print(run.download_result())
    """

    def __init__(
        self,
        *,
        training_hash: str,
        group: str,
        login: Optional[str] = None,
        password: Optional[str] = None,
        url: Optional[str] = "https://neomaril.datarisk.net/",
    ) -> None:
        super().__init__(login=login, password=password, url=url)

        self.training_hash = training_hash
        self.group = group

        url = f"{self.base_url}/training/describe/{self.group}/{self.training_hash}"
        token = refresh_token(*self.credentials, self.base_url)

        response = make_request(
            url=url,
            method="GET",
            success_code=200,
            custom_exception=TrainingError,
            custom_exception_message=f'Experiment "{training_hash}" not found.',
            specific_error_code=404,
            logger_msg=f'Experiment "{training_hash}" not found.',
            headers={
                "Authorization": f"Bearer {token}",
            },
        )

        training_data = response.json()["Description"]
        self.model_type = training_data["ModelType"]
        self.experiment_name = training_data["ExperimentName"]

    def __repr__(self) -> str:
        return f"""MLOpsTrainingExperiment(name="{self.experiment_name}", 
                                                        group="{self.group}", 
                                                        training_id="{self.training_hash}",
                                                        model_type={str(self.model_type)}
                                                        )"""

    def __str__(self):
        return f'MLOPS training experiment "{self.experiment_name} (Group: {self.group}, Id: {self.training_hash})"'

    def __describe(self):
        """
        Describe the training experiment.

        Returns
        -------
        dict
            Description of the training experiment.

        Raises
        ------
        TrainingError
            When the training can't be accessed in the server
        AuthenticationError
            Invalid credentials
        """
        url = f"{self.base_url}/v2/training/{self.training_hash}"
        token = refresh_token(*self.credentials, self.base_url)
        response = make_request(
            url=url,
            method="GET",
            success_code=200,
            custom_exception=TrainingError,
            headers={
                "Authorization": f"Bearer {token}",
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.__describe.__qualname__,
            },
        )

        return response.json()

    def project_info(self, mode="dict"):
        """
        Get the executions.

        Parameters
        ----------
        mode: str, optional
            The mode of the return value. Can be "dict" or "count". Default is "dict".

        Returns
        -------
        Union[dict, int]
            The executions in the specified mode.
        """
        describe = self.__describe()
        if mode == "dict":
            return describe
        elif mode == "count":
            return describe["ExperimentsQuantity"]
        elif mode == "log":
            yaml_data = parse_json_to_yaml(describe)
            print(yaml_data)
        else:
            raise InputError(f"Invalid mode {mode}")

    def run_training(
        self,
        *,
        run_name: str,
        training_type: str,
        description: Optional[str] = None,
        requirements_file: Optional[str] = None,
        source_file: Optional[str] = None,
        python_version: Optional[str] = "3.10",
        training_reference: Optional[str] = None,
        train_data: Optional[str] = None,
        dataset: Union[str, MLOpsDataset] = None,
        dataset_name: Optional[str] = "input",
        conf_dict: Union[dict, str] = None,
        features_file: Optional[str] = None,
        features_hash: Optional[str] = None,
        target_file: Optional[str] = None,
        target_hash: Optional[str] = None,
        output_file: Optional[str] = None,
        output_hash: Optional[str] = None,
        metrics_file: Optional[str] = None,
        parameters_file: Optional[str] = None,
        model_file: Optional[str] = None,
        extra_files: Optional[list] = None,
        env: Optional[str] = None,
        wait_complete: Optional[bool] = False,
    ) -> ITrainingExecution:
        """
        Runs a prediction from the current model.

        Parameters
        ---------
        run_name: str
            The execution name in case you'd like to search it later. Obrigatory for all training types
        training_type: str
            Either 'Custom', 'External' or 'AutoML'
        description: Optional[str], default=None
            A small description to help you remember the training
        source_file: Optional[str], default=None
            Path to the python source file. Obrigatory for 'Custom' training
        python_version: Optional[str], default='3.10'
            Version of the python executable to run. Obrigatory for 'Custom' training, optional for 'External' training
            The available options are: '3.8, '3.9' and '3.10'
        training_reference: Optional[str], default=None
            Entrypoint function name in the source file. Obrigatory for 'Custom' training
        requirements_file: Optional[str], default=None
            The requirements.txt file. Obrigatory for 'Custom' training if you wish to install dependencies
        train_data: Optional[str], default=None
            Data used to train the model. Obrigatory for 'Custom' and 'AutoML' training types
        dataset_name: Optional[str], default="input"
            Provided name the input file. Obrigatory for 'Custom' and 'AutoML' training types
        dataset: Union[str, MLOpsDataset], default=None
            Dataset generated or uploaded from other MLOps modules. It is a string, it must be the dataset hash. You can
            also provide the entire dataset class
        conf_dict: Optional[dict, str], default=None
            Configuration file or dict. It's obrigatory for 'AutoML' training. You'll provide necessary configuration
            for dealing with your data and how to train the model
        features_file: Optional[str], default=None
            Path to the features file. Obrigatory for 'External' training
        features_hash: Optional[str], default=None
            Dataset hash that will be used as features. Obrigatory for 'External' training
        target_file: Optional[str], default=None
            Path to the target file. Obrigatory for 'External' training
        target_hash: Optional[str], default=None
            Dataset hash that will be used as target. Obrigatory for 'External' training
        output_file: Optional[str], default=None
            Path to the output file. Obrigatory for 'External' training
        output_hash: Optional[str], default=None
            Dataset hash that will be used as output. Obrigatory for 'External' training
        metrics_file: Optional[str], default=None
            Path to the metrics file. Obrigatory for 'External' training
        parameters_file: Optional[str], default=None
            Path to the parameter file. Obrigatory for 'External' training
        model_file: Optional[str], default=None
            Path to the model file. Obrigatory for 'External' training
        extra_files: Optional[list], default=None
            An optional list with a path of files used to train your model
        env: Optional[str], default=None
            An optional path to the provided environment variables
        wait_complete: Optional[bool], default=False
            Lock your script/cell until training is complete.
        Raises
        ------
        AuthenticationError
        InputError

        Returns
        -------
        Example
        -------
        >>> execution = run = training.run_training(run_name=,training_type=,requirements_file=data_path+'requirements.txt',python_version='3.9',training_reference='train_model',wait_complete=True)
        """

        if training_type not in ("Custom", "External", "AutoML"):
            raise InputError(
                "Training type needs be: 'Custom', 'AutoML' or 'External'."
            )

        if dataset is None and train_data is None and training_type != "External":
            raise InputError(
                "You must provide a data to train your model. It can be a path to a file or a dataset hash."
            )

        if description is None:
            description = f"Training is {training_type}"

        if extra_files is None:
            extra_files = []

        if dataset is not None:
            dataset_hash = validate_dataset(dataset)
        else:
            dataset_hash = None

        input_data, upload_data = parse_data(
            file_path=train_data,
            form_data="dataset_hash" if dataset_hash is not None else "dataset_name",
            file_name=dataset_name,
            file_form="input",
            dataset_hash=dataset_hash,
        )

        builder = {
            "Custom": (
                CustomTrainingExecution,
                {
                    "training_hash": self.training_hash,
                    "model_type": training_type,
                    "group": self.group,
                    "login": self.credentials[0],
                    "password": self.credentials[1],
                    "url": self.base_url,
                    "run_name": run_name,
                    "description": description,
                    "input_data": input_data,
                    "upload_data": upload_data,
                    "requirements_file": requirements_file,
                    "source_file": source_file,
                    "python_version": python_version,
                    "training_reference": training_reference,
                    "extra_files": extra_files,
                    "env": env,
                    "wait_complete": wait_complete,
                },
            ),
            "AutoML": (
                AutoMLTrainingExecution,
                {
                    "training_hash": self.training_hash,
                    "model_type": training_type,
                    "group": self.group,
                    "login": self.credentials[0],
                    "password": self.credentials[1],
                    "url": self.base_url,
                    "run_name": run_name,
                    "description": description,
                    "input_data": input_data,
                    "upload_data": upload_data,
                    "conf_dict": conf_dict,
                    "extra_files": extra_files,
                    "wait_complete": wait_complete,
                },
            ),
            "External": (
                ExternalTrainingExecution,
                {
                    "training_hash": self.training_hash,
                    "model_type": training_type,
                    "group": self.group,
                    "login": self.credentials[0],
                    "password": self.credentials[1],
                    "url": self.base_url,
                    "run_name": run_name,
                    "python_version": python_version,
                    "description": description,
                    "features_file": features_file,
                    "features_hash": features_hash,
                    "target_file": target_file,
                    "target_hash": target_hash,
                    "output_file": output_file,
                    "output_hash": output_hash,
                    "metrics_file": metrics_file,
                    "parameters_file": parameters_file,
                    "model_file": model_file,
                    "requirements_file": requirements_file,
                    "wait_complete": wait_complete,
                },
            ),
        }

        builder_train_class, params = builder[training_type]
        train = builder_train_class(**params)
        return train

    def get_training_execution(self, exec_id: Optional[str] = None):
        """
        Get the execution instance.

        Parameters
        ---------
        exec_id: Optional[str], optional
            Execution id. If not informed we get the last execution.

        Returns
        -------
        MLOpsExecution
            The chosen execution
        """
        raise NotImplementedError("Get training execution not implemented.")

    @contextmanager
    def log_train(
        self,
        *,
        name,
        X_train,
        y_train,
        description: Optional[str] = None,
        save_path: Optional[str] = None,
    ):
        """
        Creates a context manager that logs training progress.

        name: str
            Run name
        X_train: DataFrame
            Features
        y_train: DataFrame
            Target
        description: str, default=None
            Description
        save_path: str, default=None
            Path to save the trained model

        Returns
        -------
        MLOpsTrainingLogger
        """
        try:
            self.trainer = MLOpsTrainingLogger(
                name=name,
                X_train=X_train,
                y_train=y_train,
                save_path=save_path,
            )
            yield self.trainer

        finally:
            self.trainer._processing_logging_inputs()
            self.run_training(
                run_name=self.trainer.name,
                training_type="External",
                features_file=self.trainer.features_file,
                target_file=self.trainer.target_file,
                output_file=self.trainer.output_file,
                metrics_file=self.trainer.metrics_file,
                parameters_file=self.trainer.params_file,
                model_file=self.trainer.model_file,
                requirements_file=self.trainer.requirements,
                description=description,
                python_version=self.trainer.python_version,
                extra_files=self.trainer.extras,
            )
            logger.info(
                "Use the `get_training_execution()` method to get a training execution."
            )


class MLOpsTrainingClient(BaseMLOpsClient):
    """
    Class for client for accessing MLOps and manage models

    Parameters
    ----------
    login: str
        Login for authenticating with the client. You can also use the env variable MLOPS_USER to set this
    password: str
        Password for authenticating with the client. You can also use the env variable MLOPS_PASSWORD to set this
    url: str
        URL to MLOps Server. The default value is https://neomaril.datarisk.net, use it to test your deployment first before changing to production. You can also use the env variable MLOPS_URL to set this

    Raises
    ------
    AuthenticationError
        Invalid credentials
    ServerError
        Server unavailable
    """

    def __repr__(self) -> str:
        return f"Codex version {constants.CODEX_VERSION}"

    def __str__(self):
        return f"Codex version {constants.CODEX_VERSION}"

    def list(self, mode="dict"):
        url = f"{self.base_url}/v2/training"
        token = refresh_token(*self.credentials, self.base_url)
        response = make_request(
            url=url,
            method="GET",
            success_code=200,
            headers={
                "Authorization": f"Bearer {token}",
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.list.__qualname__,
            },
        )

        if mode == "dict":
            return response.json()
        if mode == "count":
            return len(response.json())
        if mode == "log":
            yaml_data = parse_json_to_yaml(response.json())
            print(yaml_data)
            return
        raise InputError(f'{mode} is invalid. The options are "count", "dict" or "log"')

    def get_training(
        self, *, training_hash: str, group: str
    ) -> MLOpsTrainingExperiment:
        """
        Acess a model using its id

        Parameters
        ---------
        training_hash: str
            Training id (hash) that needs to be acessed
        group: str
            Group the model is inserted.

        Raises
        ------
        TrainingError
            Model unavailable
        ServerError
            Unknown return from server

        Returns
        -------
        MLOpsTrainingExperiment
            A MLOpsTrainingExperiment instance with the training hash from `training_id`

        Example
        -------
        >>> client = MLOpsTrainingClient()
        >>> training = client.get_training('<TRAINING_HASH>', '<GROUP>')
        """

        return MLOpsTrainingExperiment(
            training_hash=training_hash,
            login=self.credentials[0],
            password=self.credentials[1],
            group=group,
            url=self.base_url,
        )

    def __get_repeated_thash(
        self, model_type: str, experiment_name: str, group: str
    ) -> Union[str, None]:
        """Look for a previous train experiment.

        Args:
            experiment_name (str): name given to the training, should be not null, case-sensitive, have between 3 and 32 characters,
                                   that could be alphanumeric including accentuation (for example: 'é', à', 'ç','ñ') and space,
                                   without blank spaces and special characters

            model_type (str): type of the model being trained. It can be
                                Classification: for ML algorithms related to classification (predicts discrete class labels) problems;
                                Regression: the ones that will use regression (predict a continuous quantity) algorithms;
                                Unsupervised: for training that will use ML algorithms without supervision.

            group (str): name of the group, previously created, where the training will be inserted

        Raises:
            InputError: some part of the data is incorrect
            AuthenticationError: user has insufficient permissions
            ServerError: server is not available
            Exception: generated exception in case of the response to the request is different from 201

        Returns:
            str | None: THash if it is found, otherwise, None is returned
        """
        url = f"{self.base_url}/training/search"
        token = refresh_token(*self.credentials, self.base_url)

        response = make_request(
            url=url,
            method="GET",
            success_code=200,
            headers={
                "Authorization": f"Bearer {token}",
            },
        )

        results = response.json().get("Results")
        for result in results:
            condition = (
                result["ExperimentName"] == experiment_name
                and result["GroupName"] == group
                and result["ModelType"] == model_type
            )
            if condition:
                logger.info("Found experiment with same attributes...")
                return result["TrainingHash"]

    def __register_training(
        self, experiment_name: str, model_type: str, group: str
    ) -> str:
        """Creates a train experiment. A train experiment can aggregate multiple training runs (also called executions).
        Each execution can eventually become a deployed model or not.

        Args:
            experiment_name (str): name given to the training, should be not null, case-sensitive, have between 3 and 32 characters,
                                   that could be alphanumeric including accentuation (for example: 'é', à', 'ç','ñ') and space,
                                   without blank spaces and special characters

            model_type (str): type of the model being trained. It can be
                                Classification: for ML algorithms related to classification (predicts discrete class labels) problems;
                                Regression: the ones that will use regression (predict a continuous quantity) algorithms;
                                Unsupervised: for training that will use ML algorithms without supervision.

            group (str): name of the group, previous created, where the training will be inserted

        Raises:
            InputError: some part of the data is incorrect
            AuthenticationError: user has insufficient permissions
            ServerError: server is not available
            Exception: generated exception in case of the response to the request is different from 201

        Returns:
            str: training hash of the experiment
        """
        url = f"{self.base_url}/v2/training/{group}"
        token = refresh_token(*self.credentials, self.base_url)

        payload = {"Name": experiment_name, "Type": model_type}

        response = make_request(
            url=url,
            method="POST",
            success_code=201,
            json=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Neomaril-Origin": "Codex",
                "Neomaril-Method": self.__register_training.__qualname__,
            },
        )

        response_data = response.json()
        logger.info(response_data["Message"])
        training_hash = response_data["TrainingHash"]
        return training_hash

    def create_training_experiment(
        self,
        *,
        experiment_name: str,
        model_type: str,
        group: str,
        force: Optional[bool] = False,
    ) -> MLOpsTrainingExperiment:
        """
        Create a new training experiment on MLOps.

        Parameters
        ---------
        experiment_name: str
            The name of the experiment, in less than 32 characters
        model_type: str
            The name of the scoring function inside the source file.
        group: str
            Group the model is inserted. Default to 'datarisk' (public group)
        force: Optional[bool], optional
            Forces to create a new training with the same model_type, experiment_name, group

        Raises
        ------
        InputError
            Some input parameters its invalid
        ServerError
            Unknow internal server error

        Returns
        -------
        MLOpsTrainingExperiment
            A MLOpsTrainingExperiment instance with the training hash from `training_id`

        Example
        -------
        >>> training = client.create_training_experiment('Training example', 'Classification', 'ex_group')
        """

        validate_group_existence(group, self)

        if model_type not in ["Classification", "Regression", "Unsupervised"]:
            raise InputError(
                f"Invalid model_type {model_type}. Should be one of the following: Classification, Regression or "
                f"Unsupervised"
            )

        logger.info("Trying to load experiment...")
        training_hash = self.__get_repeated_thash(
            model_type=model_type, experiment_name=experiment_name, group=group
        )

        if force or training_hash is None:
            msg = (
                "The experiment you're creating has identical name, group, and model type attributes to an existing one. "
                + "Since forced creation is active, we will continue with the process as specified"
                if force
                else "Could not find experiment. Creating a new one..."
            )
            logger.info(msg)
            training_hash = self.__register_training(
                experiment_name=experiment_name, model_type=model_type, group=group
            )

        return MLOpsTrainingExperiment(
            training_hash=training_hash,
            login=self.credentials[0],
            password=self.credentials[1],
            group=group,
            url=self.base_url,
        )
