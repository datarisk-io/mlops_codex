#!/usr/bin/env python
# coding: utf-8

import io
import re
from typing import Union, Optional
import requests
import json
from time import sleep
from neomaril_codex._base import *
from neomaril_codex.model import NeomarilModel
from neomaril_codex.exceptions import *

patt = re.compile(r'(\d+)')

class NeomarilTrainingExecution(NeomarilExecution):
    """
    Class to manage trained models.

    Arguments
    ---------
    training_id : str
        Training id (hash) from the experiment you want to access
    group : str
        Group the training is inserted. Default is 'datarisk' (public group)
    exec_id : str
        Executiong id for that especific training run
    password : str
        Password for authenticating with the client
    environment : str
        Enviroment of Neomaril you are using. 

    Raises
    ------
    TrainingError
        When the training can't be acessed in the server
    AuthenticationError
        Unvalid credentials
    """

    def __init__(self, training_id:str, group:str, exec_id:str, password:Optional[str]=None, url:str=None) -> None:

        super().__init__(training_id, 'Training', exec_id=exec_id, password=password, url=url, group=group)
        load_dotenv()
        self.__credentials = os.getenv('NEOMARIL_TOKEN') if os.getenv('NEOMARIL_TOKEN') else password

        self.training_id = training_id
        self.group = group
    
        self.training_type = self.execution_data['TrainingType']

    def __upload_model(self, model_name:str, model_reference:Optional[str]=None, source_file:Optional[str]=None, 
                                         schema:Optional[Union[str, dict]]=None, extra_files:Optional[list]=None, 
                                         env:Optional[str]=None, operation:str='Sync', input_type:str=None) -> str:
        """
        Upload the files to the server

        Arguments
        ---------
        model_name : str
            The name of the model, in less than 32 characters
        model_reference : str, optional
            The name of the scoring function inside the source file
        source_file : str, optional
            Path of the source file. The file must have a scoring function that accepts two parameters: data (data for the request body of the model) and model_path (absolute path of where the file is located)
        schema : Union[str, dict], optional
            Path to a JSON or XML file with a sample of the input for the entrypoint function. A dict with the sample input can be send as well
        extra_files list, optional
            A optional list with additional files paths that should be uploaded. If the scoring function refer to this file they will be on the same folder as the source file
        env : str, optional
            Flag that choose which environment (dev, staging, production) of Neomaril you are using. Default is True
        operation : str
            Defines wich kind operation is beeing executed (Sync or Async). Default value is Sync
        input_type : str
            The type of the input file that should be 'json', 'csv' or 'parquet'

        Raises
        ------
        InputError
            Some input parameters its invalid

        Returns
        -------
        str
            The new model id (hash)
        """
        
        url = f"{self.base_url}/training/promote/{self.group}/{self.training_id}/{self.exec_id}"

        form_data = {'name': model_name, 'operation': operation}
        upload_data = []

        if self.training_type == 'Custom':
            form_data['model_reference'] = model_reference
            form_data['input_type'] = input_type
        
            file_extesions = {'py': 'script.py', 'ipynb': "notebook.ipynb"}
        

            upload_data = [
                ("source", (file_extesions[source_file.split('.')[-1]], open(source_file, "r")))
            ]

            if env:
                upload_data.append(("env", (".env", env)))
            
            if extra_files:
                extra_data = [('extra', (c.split('/')[-1], open(c, "r"))) for c in extra_files]
                
                upload_data += extra_data
        
        else:

            input_type = 'automl'

        if operation=="Sync":
            input_type = "json"
            if schema:
                if isinstance(schema, str):
                    schema_file = open(schema, "r")
                elif isinstance(schema, dict):
                    schema_file = io.StringIO()
                    json.dump(schema, schema_file).seek(0)
                upload_data.append(("schema", ("schema.json", schema_file)))
            else:
                raise InputError("Schema file is mandatory for Sync models")

        else:
            if input_type == 'json|csv|parquet':
                raise InputError("Choose a input type from "+input_type)

            
        response = requests.post(url, data=form_data, files=upload_data, headers={'Authorization': 'Bearer ' + self.__credentials})
        
        if response.status_code == 201:
            data = response.json()
            model_id = data["ModelHash"]
            logger.info(f'{data["Message"]} - Hash: "{model_id}"')
            return model_id
        else:
            logger.error('Upload error: ' + response.text)
            raise InputError('Invalid parameters for model creation')

    def __host_model(self, operation:str, model_id:str) -> None:
        """
        Builds the model execution environment

        Arguments
        ----------
        operation : str
            The model operation type (Sync or Async)
        model_id : str
            The uploaded model id (hash)

        Raises
        ------
        InputError
            Some input parameters its invalid
        """
        
        url = f"{self.base_url}/model/{operation}/host/{self.group}/{model_id}"
        if operation == 'sync':
            url = url.replace('7070','7071')
        response = requests.get(url, headers={'Authorization': 'Bearer ' + self.__credentials})
        if response.status_code == 202:
            logger.info(f"Model host in process - Hash: {model_id}")
        else:
            logger.error(response.text)
            raise InputError('Invalid parameters for model creation')

    def promote_model(self, model_name:str, model_reference:Optional[str]=None, source_file:Optional[str]=None, 
                                         schema:Optional[Union[str, dict]]=None, extra_files:Optional[list]=None, 
                                         env:Optional[str]=None, operation:str='Sync', input_type:str=None)-> NeomarilModel:
        """
        Upload models trained inside Neomaril.

        Arguments
        ---------
        model_name : str
            The name of the model, in less than 32 characters
        model_reference : str, optional
            The name of the scoring function inside the source file
        source_file : str, optional
            Path of the source file. The file must have a scoring function that accepts two parameters: data (data for the request body of the model) and model_path (absolute path of where the file is located)
        schema : Union[str, dict], optional
            Path to a JSON or XML file with a sample of the input for the entrypoint function. A dict with the sample input can be send as well
        extra_files list, optional
            A optional list with additional files paths that should be uploaded. If the scoring function refer to this file they will be on the same folder as the source file
        env : str, optional
            Flag that choose which environment (dev, staging, production) of Neomaril you are using. Default is True
        operation : str
            Defines wich kind operation is beeing executed (Sync or Async). Default value is Sync
        input_type : str
            The type of the input file that should be 'json', 'csv' or 'parquet'

        Raises
        ------
        TrainingError
            The training execution shouldn't be succeeded to be promoted

        Returns
        -------
        NeomarilModel
            The new training model
        """
        if self.status in ['Running', 'Requested']:
            self.status = self.get_status()['Status']
        
        if self.status != 'Succeeded':
            raise TrainingError("Training execution must be Succeeded to be promoted, current status is "+self.status)
            
        model_id = self.__upload_model(model_name, model_reference=model_reference, source_file=source_file, env=env,
                                                                        schema=schema, extra_files=extra_files, operation=operation, 
                                                                        input_type=input_type)
        
        if model_id:
            self.__host_model(operation.lower(), model_id)

            return NeomarilModel(model_id, password=self.__credentials, group=self.group, url=self.base_url)

        
class NeomarilTrainingExperiment(BaseNeomaril):
    """
    Class to manage models being trained inside Neomaril

    Arguments
    ---------
    password : str
        Password for authenticating with the client
    training_id : str
        Training id (hash) from the experiment you want to access
    group : str
        Group the training is inserted. Default is 'datarisk' (public group)
    environment : str
        Flag that choose which environment of Neomaril you are using. Test your deployment first before changing to production. Default is True

    Raises
    ------
    TrainingError
        When the training can't be acessed in the server
    AuthenticationError
        Unvalid credentials
    """

    def __init__(self, training_id:str, password:Optional[str]=None, group:str="datarisk", url:str='https://neomaril.staging.datarisk.net/') -> None:

        super().__init__()
        load_dotenv()
        self.__credentials = os.getenv('NEOMARIL_TOKEN') if os.getenv('NEOMARIL_TOKEN') else password
        self.training_id = training_id
        self.base_url = os.getenv('NEOMARIL_URL') if os.getenv('NEOMARIL_URL') else url
        self.base_url = parse_url(self.base_url)
        self.group = group

        try_login(self.__credentials, self.base_url)
        
        url = f"{self.base_url}/training/describe/{self.group}/{self.training_id}"
        response = requests.get(url, headers={'Authorization': 'Bearer ' + self.__credentials})
    
        if response.status_code == 404:
            raise ModelError(f'Experiment "{training_id}" not found.')
            
        elif response.status_code >= 500:
            raise ModelError(f'Unable to retrive experiment "{training_id}"')
    
        self.training_data = response.json()['Description']
        self.model_type = self.training_data['ModelType']
        self.training_type = self.training_data['TrainingType']
        self.experiment_name = self.training_data['ExperimentName']

    def __repr__(self) -> str:
            return f"""NeomarilTrainingExperiment(name="{self.experiment_name}", 
                                                        group="{self.group}", 
                                                        training_id="{self.training_id}",
                                                        training_type="{self.training_type}",
                                                        model_type={str(self.model_type)}
                                                        )"""

    def __str__(self):
        return f'NEOMARIL training experiment "{self.experiment_name} (Group: {self.group}, Id: {self.training_id})"'
    
    
    def __upload_training(self, run_name:str, train_data:str, training_reference:Optional[str]=None, 
                                                python_version:str='3.8', conf_dict:Optional[Union[str, dict]]=None,
                                                source_file:Optional[str]=None, requirements_file:Optional[str]=None,
                                                extra_files:Optional[list]=None) -> str:
        
        """
        Upload the files to the server

        Arguments
        ---------
        run_name : str
            The name of the model, in less than 32 characters
        train_data : str
            Path of the file with train data
        training_reference : str, optional
            The name of the training function inside the source file. Just used when training_type is Custom
        python_version : str
            Python version for the model environment. Avaliable versions are 3.7, 3.8, 3.9, 3.10. Defaults to '3.8'. Just used when training_type is Custom
        conf_dict : Union[str, dict], optional
            Path to a JSON file with a the AutoML configuration. A dict can be send as well. Just used when training_type is AutoML
        source_file : str, optional
            Path of the source file. The file must have a training function that accepts one parameter: model_path (absolute path of where the file is located). Just used when training_type is Custom
        requirements_file : str, optional
            Path of the requirements file. The packages versions must be fixed eg: pandas==1.0. Just used when training_type is Custom
        extra_files : list, optional
            A optional list with additional files paths that should be uploaded. If the scoring function refer to this file they will be on the same folder as the source file. Just used when training_type is Custom
        
        Raises
        ------
        InputError
            Some input parameters its invalid

        Returns
        -------
        str
            The new model id (hash)
        """
        
        url = f"{self.base_url}/training/upload/{self.group}/{self.training_id}"

        upload_data = [
                ("train_data", (train_data.split('/')[-1], open(train_data, "r")))
            ]

        if self.training_type == 'Custom':
        
            file_extesions = {'py': 'app.py', 'ipynb': "notebook.ipynb"}
        
            upload_data = upload_data + [
                ("source", (file_extesions[source_file.split('.')[-1]], open(source_file, "r"))),
                ("requirements", ("requirements.txt", open(requirements_file, "r")))
            ]
         
            if extra_files:
                extra_data = [('extra', (c.split('/')[-1], open(c, "r"))) for c in extra_files]
                
                upload_data += extra_data
                
            form_data = {'run_name': run_name, 'training_reference': training_reference,
                                    'python_version': "Python"+python_version.replace('.', '')}
        
        elif self.training_type == 'AutoML':
                
            form_data = {'run_name': run_name}

            if conf_dict:
                if isinstance(conf_dict, str):
                    schema_file = open(conf_dict, "r")
                elif isinstance(conf_dict, dict):
                    schema_file = io.StringIO()
                    json.dump(conf_dict, schema_file).seek(0)
                upload_data.append(("conf_dict", ("conf.json", schema_file)))
            else:
                raise InputError("conf_dict is mandatory for AutoML training")

        response = requests.post(url, data=form_data, files=upload_data, headers={'Authorization': 'Bearer ' + self.__credentials})
        
        message = response.text

        if response.status_code == 201:
            logger.info(message)
            return re.search(patt, message).group(1)
        else:
            logger.error(message)
            raise InputError('Bad input for training upload')

    def __execute_training(self, exec_id:str) -> None:
        """
        Builds the model execution environment

        Arguments
        ---------
        exec_id : str
            The uploaded training execution id (hash)

        Raises
        ------
        InputError
            Some input parameters its invalid
        """
        
        url = f"{self.base_url}/training/execute/{self.group}/{self.training_id}/{exec_id}"
        response = requests.get(url, headers={'Authorization': 'Bearer ' + self.__credentials})
        if response.status_code == 200:
            logger.info(f"Model training starting - Hash: {self.training_id}")
        else:
            logger.error(response.text)
            raise InputError('Invalid parameters for training execution')

    def run_training(self, run_name:str, train_data:str, training_reference:Optional[str]=None, 
                                                python_version:str='3.8', conf_dict:Optional[Union[str, dict]]=None,
                                                source_file:Optional[str]=None, requirements_file:Optional[str]=None,
                                                extra_files:Optional[list]=None, wait_complete:Optional[bool]=False) -> Union[dict, NeomarilExecution]:
        """
        Runs a prediction from the current model.

        Arguments
        run_name : str
            The name of the model, in less than 32 characters
        train_data : str
            Path of the file with train data.
        training_reference : str, optional
            The name of the training function inside the source file. Just used when training_type is Custom
        python_version : str, optional
            Python version for the model environment. Avaliable versions are 3.7, 3.8, 3.9, 3.10. Defaults to '3.8'. Just used when training_type is Custom
        conf_dict : Union[str, dict]
            Path to a JSON file with a the AutoML configuration. A dict can be send as well. Just used when training_type is AutoML
        source_file : str, optional
            Path of the source file. The file must have a training function that accepts one parameter: model_path (absolute path of where the file is located). Just used when training_type is Custom
        requirements_file : str
            Path of the requirements file. The packages versions must be fixed eg: pandas==1.0. Just used when training_type is Custom
        extra_files : list, optional
            A optional list with additional files paths that should be uploaded. If the scoring function refer to this file they will be on the same folder as the source file. Just used when training_type is Custom
        wait_complete : bool, optional
            Boolean that informs if a model training is completed (True) or not (False). Default value is False
        
        Raises
        ------
        InputError
            Some input parameters its invalid

        Returns
        -------
        Union[dict, NeomarilExecution]
            The return of the scoring function in the source file for Sync models or the execution class for Async models.
        """
        if python_version not in ['3.7', '3.8', '3.9', '3.10']:
            raise InputError('Invalid python version. Avaliable versions are 3.7, 3.8, 3.9, 3.10')

        if self.training_type == 'Custom':
            exec_id = self.__upload_training(run_name, train_data, training_reference=training_reference,
                                                                            python_version=python_version, source_file=source_file,
                                                                            requirements_file=requirements_file, extra_files=extra_files)

        elif self.training_type == 'AutoML':
            exec_id = self.__upload_training(run_name, train_data, conf_dict=conf_dict)

        else:
            raise InputError('Invalid training type')

        if exec_id:
            self.__execute_training(exec_id)
            run = NeomarilTrainingExecution(self.training_id, self.group, exec_id, password=self.__credentials, url=self.base_url)
            status = run.get_status()['Status']
            if wait_complete:
                print('Wating the training run.', end='')
                while status in ['Running', 'Requested']:
                    sleep(30)
                    print('.', end='', flush=True)
                    status = run.get_status()['Status']
            return run

    def __call__(self, data: dict) -> dict:
            return self.predict(data)

    def get_training_execution(self, exec_id:str) -> None:
        """
        Get a execution instace.

        Arguments
        ---------
        exec_id : str
            Execution id

        Returns
        -------
        NeomarilExecution
            The new execution
        """
        return NeomarilTrainingExecution(self.training_id, self.group, exec_id, password=self.__credentials, url=self.base_url)

class NeomarilTrainingClient(BaseNeomarilClient):
    """
        Class for client for acessing Neomaril and manage models

    Arguments
    ---------
    password : str
        Password for authentication with the client
	url : str
		URL to Neomaril Server. Default value is staging, use it to test your deployment first before changing to production
        
    Raises
    ------
    AuthenticationError
        Unvalid credentials
    ServerError
        Server unavailable
    """
    def __init__(self, password:Optional[str]=None, url:str='https://neomaril.staging.datarisk.net/') -> None:
        """Client for acessing Neomaril and manage models

        Args:
                password (str): Password for authenticating with the client
            url (str): URL for Neomaril server. Test your deployment first before changing to production. Default is https://neomaril.staging.datarisk.net/

        Raises:
                AuthenticationError: Unvalid credentials
                ServerError: Server unavailable
        """
        load_dotenv()

        self.__credentials = os.getenv('NEOMARIL_TOKEN') if os.getenv('NEOMARIL_TOKEN') else password
        self.base_url = os.getenv('NEOMARIL_URL') if os.getenv('NEOMARIL_URL') else url
        self.base_url = parse_url(self.base_url)

        super().__init__(password=self.__credentials, url=self.base_url)
            
    def __repr__(self) -> str:
            return f'NeomarilTrainingClient(url="{self.base_url}", version="{self.client_version}")'
        
    def __str__(self):
        return f"NEOMARIL {self.base_url} Training client:{self.client_version}"
        
    
    def get_training(self, training_id:str, group:str="datarisk") -> NeomarilTrainingExperiment:
        """
        Acess a model using its id

        Arguments
        ---------
        training_id : str
            Training id (hash) that needs to be acessed
        group : str
            Group the model is inserted. Default is 'datarisk' (public group)

        Raises
        ------
        TrainingError
            Model unavailable
        ServerError
            Unknown return from server

        Returns
        -------
        NeomarilTrainingExperiment
            A NeomarilTrainingExperiment instance with the training hash from `training_id`
        """

        return NeomarilTrainingExperiment(training_id, password=self.__credentials, group=group)
    

    def create_training_experiment(self, experiment_name:str, model_type:str, training_type:str, group:str='datarisk')-> NeomarilTrainingExperiment:
        """
        Create a new training experiment on Neomaril.

        Arguments
        ---------
        experiment_name : str
            The name of the experiment, in less than 32 characters
        model_type : str
            The name of the scoring function inside the source file.
        training_type : str
            Path of the source file. The file must have a scoring function that accepts two parameters: data (data for the request body of the model) and model_path (absolute path of where the file is located)
        group : str
            Group the model is inserted. Default to 'datarisk' (public group)

        Raises
        ------
        InputError
            Some input parameters its invalid
        ServerError
            Unknow internal server error

        Returns
        -------
        NeomarilTrainingExperiment
            A NeomarilTrainingExperiment instance with the training hash from `training_id`
        """
        
        
        if group:
            group = group.lower().strip().replace(" ", "_").replace(".", "_").replace("-", "_")

            groups = [g["Name"] for g in self.list_groups()]

            if group not in groups:

                raise GroupError('Group dont exist. Create a group first.')

        else:
            group = 'datarisk'
            logger.info("Group not informed, using default 'datarisk' group")

        if model_type not in ['Classification', 'Regression', 'Unsupervised']:
            raise InputError(f'Invalid model_type {model_type}. Should be one of the following: Classification, Regression or Unsupervised')

        if training_type not in ['Custom', 'AutoML']:
            raise InputError(f'Invalid training_type {training_type}. Should be one of the following: Custom or AutoML')

        url = f"{self.base_url}/training/register/{group}"

        data = {'experiment_name': experiment_name, 'model_type': model_type, 'training_type': training_type}

        response = requests.post(url, data=data,
                                                         headers={'Authorization': 'Bearer ' + self.__credentials})

        if response.status_code == 200:
            message = response.json()['Message']
            logger.info(message)
            training_id = message.replace('New Training inserted with hash ', '').replace('.', '')
        else:
            logger.error(response.text)
            raise ServerError('')

        
        return NeomarilTrainingExperiment(training_id, password=self.__credentials, group=group)  