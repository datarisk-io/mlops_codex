#!/usr/bin/env python
# coding: utf-8

import io, os
from typing import Union, Optional
from time import sleep
import requests
import json
from neomaril_codex._base import *
from neomaril_codex.exceptions import *

class NeomarilModel(BaseNeomaril):
    """ Class to manage Models deployed inside Neomaril
    """

    def __init__(self, password:str, model_id:str, group:str="datarisk", group_token:Optional[str]=None, enviroment:str='staging') -> None:
        """Class to manage Models deployed inside Neomaril

        Args:
            password (str): Password for authenticating with the client
            model_id (str): Model id (hash) from the model you want to acess
            group (str): Group the model is inserted. Default is 'datarisk' (public group)
            group_token (str): Token for executing the model (show when creating a group). It can be informed when getting the model or when running predictions
            enviroment (str): Flag that choose which enviroment of Neomaril you are using. Test your deployment first before changing to production. Default is True

        Raises:
            ModelError: When the model can't be acessed in the server
            AuthenticationError: Unvalid credentials
        """
        super().__init__()
        self.__credentials = password
        self.model_id = model_id
        self.group = group
        self.__token = group_token

        if self.enviroment == "dev":
            self.base_url = self._dev_url
        elif self.enviroment == 'staging':
            self.base_url = self._staging_url
        elif self.enviroment == "production":
            self.base_url = self._production_url

        try_login(self.__credentials, self.base_url)
        
        url = f"{self.base_url}/model/describe/{self.group}/{self.model_id}"
        response = requests.get(url, headers={'Authorization': 'Bearer ' + self.__credentials})
    
        if response.status_code == 404:
            raise ModelError(f'Model "{model_id}" not found.')
    
        
        elif response.status_code >= 500:
            raise ModelError(f'Unable to retrive model "{model_id}"')
    
        self.model_data = response.json()['Description']
        self.name = self.model_data['Name']
        self.status = self.model_data['Status']
        self.schema = self.model_data['Schema']
        self.operation = self.model_data['Operation'].lower()
        self.__model_ready = self.status == "Deployed"

    def __repr__(self) -> str:
        return f"""NeomarilModel(name="{self.name}", group="{self.group}", 
                                status="{self.status}", enviroment="{self.enviroment}"
                                model_id="{self.model_id}",
                                operation="{self.operation.title()}",
                                schema={str(self.schema)}
                                )"""

    def __str__(self):
        return f'NEOMARIL model "{self.name} (Group: {self.group}, Id: {self.model_id})"'

    def __get_status(self):
        url = f"{self.base_url}/model/status/{self.group}/{self.model_id}"
        response = requests.get(url, headers={'Authorization': 'Bearer ' + self.__credentials})
        if response.status_code < 300:
            return response.json()['Status']
        else:
            raise ModelError(response.text)

    def wait_ready(self):
        if self.status in ['Ready', 'Building']:
            self.status = self.__get_status()
            while self.status == 'Building':
                sleep(30)
                self.status = self.__get_status()


    def health(self):
        if self.operation == 'async':
            try:
                try_login(self.__credentials, self.base_url)
                return 'OK'
            except Exception as e:
                logger.error('Server error: '+e)
                return 'NOK'
        elif self.operation == 'sync':
            url = f"{self.base_url.replace('7070', '7071')}/model/sync/health/{self.group}/{self.model_id}"
            response = requests.get(url, headers={'Authorization': 'Bearer ' + self.__credentials})
            if response.status_code == 200:
                return response.json()['Message']
            else:
                logger.error('Server error: '+response.text)
                return 'NOK'


    def restart_model(self, wait_for_ready:bool=True):
        if (self.operation == "sync") and (self.status == "Deployed"):
            url = f"{self.base_url.replace('7070', '7071')}/model/sync/restart/{self.group}/{self.model_id}"
            response = requests.get(url, headers={'Authorization': 'Bearer ' + self.__credentials})
            if response.status_code < 300:
                logger.info("Model is restarting")
                self.status = self.__get_status()
                if wait_for_ready:
                    print('Wating for deploy to be ready.', end='')
                    while self.status == 'Building':
                        sleep(30)
                        self.status = self.__get_status()
                        print('.', end='', flush=True)


    def get_logs(self, start:Optional[str]=None, end:Optional[str]=None, routine:Optional[str]=None, type:Optional[str]=None):
        url = f"{self.base_url}/model/logs/{self.model_id}"
        return self._logs(url, self.__credentials, start=start, end=end, routine=routine, type=type)
    
    def delete(self):
        """Deletes the current model.
        IMPORTANT! For now this is irreversible, if you want to use the model again later you will need to upload again (and it will have a new ID).

        Raises:
            ServerError: _description_

        Returns:
            _type_: _description_
        """
        if self.__model_ready:
            req = requests.delete(f"{self.base_url}/model/delete/{self.group}/{self.model_id}", headers={'Authorization': 'Bearer ' + self.__credentials})
        
            if req.status_code == 200:
                response = requests.get(f"{self.base_url}/model/describe/{self.group}/{self.model_id}", 
                                            headers={'Authorization': 'Bearer ' + self.__credentials})
            
                self.model_data = response.json()['Description']
                self.status = self.model_data['Status']
                self.__model_ready = False
        
                return req.json()
            else:
                raise ServerError('Model deleting failed')
      
        else:
            return 'Model is '+self.status
  
    def set_token(self, group_token:str) -> None:
        """Saves the group token for this model instance.

        Args:
            group_token (str): Token for executing the model (show when creating a group)

        """

        self.__token = group_token
        logger.info(f"Token for group {self.group} added.")

    def predict(self, data:Union[dict, str], group_token:Optional[str]=None, wait_complete:Optional[bool]=False) -> Union[dict, NeomarilExecution]:
        """Runs a prediction from the current model.

        Args:
            data (dict, str): The same data that is used in the source file. 
            If Sync is a dict, the keys that are needed inside this dict are the ones in the `schema` atribute.
            If Async is a string with the file path with the same filename used in the source file. 
            group_token (str): Token for executing the model (show when creating a group). It can be informed when getting the model or when running predictions

        Raises:
            ModelError: Model is not available

        Returns:
            Union[dict, NeomarilExecution]: The return of the scoring function in the source file for Sync models or the execution class for Async models.
        """
        if self.__model_ready:
            if (group_token is not None) | (self.__token is not None):
                url = f"{self.base_url}/model/{self.operation}/run/{self.group}/{self.model_id}"
                if self.__token:
                    group_token = self.__token
                if self.operation == 'sync':
                    url = url.replace('7070', '7071')
                    model_input = {
                            "Input": data
                    }

                    req = requests.post(url, data=json.dumps(model_input), headers={'Authorization': 'Bearer ' + group_token})

                    return req.json()

                elif self.operation == 'async':

                    req = requests.post(url, files=[("input", (data.split('/')[-1], open(data, "r")))],
                                                    headers={'Authorization': 'Bearer ' + group_token})


                    if req.status_code == 202:
                        message = req.json()
                        logger.info(message['Message'])
                        exec_id = message['ExecutionId']
                        run = NeomarilExecution(self.model_id, 'AsyncModel', exec_id=exec_id, password=self.__credentials, 
                                                enviroment=self.enviroment, group=self.group)
                        status = run.get_status()['Status']
                        if wait_complete:
                            print('Wating the training run.', end='')
                            while status in ['Running', 'Requested']:
                                sleep(30)
                                print('.', end='', flush=True)
                                status = run.get_status()['Status']
                        return run
                    else:
                        raise ServerError(req.text)
                
            else:
                raise AuthenticationError("Group token not informed")
        else:
            url = f"{self.base_url}/model/describe/{self.group}/{self.model_id}"
            response = requests.get(url, headers={'Authorization': 'Bearer ' + self.__credentials}).json()['Description']
            if response['Status'] == "Deployed":
                self.model_data = response
                self.status = response['Status']
                self.__model_ready = True
                return self.predict(data)
    
            else:
                raise ModelError('Model is not available to predictions')

    def __call__(self, data: dict) -> dict:
        return self.predict(data)

    def get_model_execution(self, exec_id:str) -> None:
        """Get a execution instace for that model.

        Args:
            exec_id (str): Execution id

        Raises:
        ModelError: if the user tries to get a execution from a Sync model 

        """
        if self.operation == 'async':
            return NeomarilExecution(self.model_id, 'AsyncModel', exec_id, password=self.__credentials, 
                                     enviroment=self.enviroment, group=self.group)
        else:
            raise ModelError("Sync models don't have executions")

    def register_monitoring(self, preprocess_reference:str, shap_reference:str, configuration_file:str, preprocess_file:Optional[str]=None,
                            requirements_file:Optional[str]=None) -> None:

        url = f"{self.base_url}/monitoring/register/{self.group}/{self.model_id}"
    
        upload_data = [
            ("configuration", ('conf.json', open(configuration_file, "r"))),
        ]

        form_data = {'preprocess_reference': preprocess_reference, 'shap_reference': shap_reference}

        if preprocess_file:
            upload_data.append(("source", ('preprocess.'+preprocess_file.split('.')[-1], open(preprocess_file, "r"))))
            
            if preprocess_file.endswith('py'):
                form_data['type'] = 'PythonScript'
            elif preprocess_file.endswith('ipynb'):
                form_data['type'] = 'PythonNotebook'
        else:
            form_data['type'] = 'ModelScript'
            
        if requirements_file:
            upload_data.append(("requirements", ('requirements.txt', open(requirements_file, "r"))))

        response = requests.post(url, data=form_data, files=upload_data, headers={'Authorization': 'Bearer ' + self.__credentials})
        
        if response.status_code == 201:
            data = response.json()
            model_id = data["ModelHash"]
            logger.info(f'{data["Message"]} - Hash: "{model_id}"')
            return model_id
        else:
            logger.error('Upload error: ' + response.text)
            raise InputError('Invalid parameters for model creation')

class NeomarilModelClient(BaseNeomarilClient):
    """Client for acessing Neomaril and manage models

    """
    def __init__(self, password:str, enviroment:str='staging') -> None:
        """Client for acessing Neomaril and manage models

        Args:
                password (str): Password for authenticating with the client
                enviroment (str): Flag that choose which enviroment of Neomaril you are using. Test your deployment first before changing to production. Default is True

        Raises:
                AuthenticationError: Unvalid credentials
                ServerError: Server unavailable
        """
        super().__init__(password, enviroment=enviroment)
        self.__credentials = password
            
    def __repr__(self) -> str:
            return f'NeomarilModelClient(enviroment="{self.enviroment}", version="{self.client_version}")'
        
    def __str__(self):
        return f"NEOMARIL {self.enviroment} Model client:{self.client_version}"
        
    def __get_model_status(self, model_id:str, group:str) -> dict:
        """Gets the status of the model with the hash equal to `model_id`

        Args:
                group (str): Group the model is inserted
                model_id (str): Model id (hash) from the model being searched

        Raises:
                ModelError: Model unavailable

        Returns:
                dict: Returns the model status and a message if the status is 'Failed'.
        """

        url = f"{self.base_url}/model/status/{group}/{model_id}"
        response = requests.get(url, headers={'Authorization': 'Bearer ' + self.__credentials})
        if response.status_code not in [200, 410]:
            raise ModelError(f'Model "{model_id}" not found')
        
        return response.json()
    
    def get_model(self, model_id:str, group:str="datarisk", group_token:Optional[str]=None, wait_for_ready:bool=True) -> NeomarilModel:
        """Acess a model using its id

        Args:
                model_id (str): Model id (hash) that needs to be acessed
                group (str): Group the model is inserted. Default is 'datarisk' (public group)
                group_token (str): Token for executing the model (show when creating a group). It can be informed when getting the model or when running predictions
                wait_for_ready (bool, optional): If the model is being deployed, wait for it to be ready instead of failing the request. Defaults to True.

        Raises:
                ModelError: Model unavailable
                ServerError: Unknown return from server

        Returns:
                NeomarilModel: A NeomarilModel instance with the model hash from `model_id`
        """
        try:
            response = self.__get_model_status(model_id, group)
        except KeyError:
            raise ModelError("Model not found")
        
        status = response['Status']
        
        if status == 'Building':
            if wait_for_ready:
                print('Wating for deploy to be ready.', end='')
                while status == 'Building':
                    status = self.__get_model_status(model_id, group)['Status']
                    print('.', end='', flush=True)
                    sleep(10)
            else:
                raise ModelError(f'Model "{model_id}" not ready yet')
            
        if status in ['Disabled', 'Ready']:
            raise ModelError(f'Model "{model_id}" unavailable (disabled or deploy process is incomplete)')
        elif status == 'Failed':
            logger.error(str(response['Message']))
            raise ModelError(f'Model "{model_id}" deploy failed, so model is unavailable.')
        elif status == 'Deployed': 
            logger.info(f'Model {model_id} its deployed. Fetching model.')
            return NeomarilModel(self.__credentials, model_id, group=group, enviroment=self.enviroment, group_token=group_token)
        else:
            raise ServerError('Unknown model status: ',status)
    
    def search_models(self, name:Optional[str]=None, state:Optional[str]=None, 
                                        group:Optional[str]=None, only_deployed:bool=False) -> list:
        """Search for models using the name of the model

        Args:
                name (Optional[str]): Text that its expected to be on the model name. It runs similar to a LIKE query on SQL.
                group (Optional[str]): Text that its expected to be on the group name. It runs similar to a LIKE query on SQL.
                state (Optional[str]): Text that its expected to be on the state. It runs similar to a LIKE query on SQL.
                only_deployed (bool, optional): If its True, filter only models ready to be used (status == "Deployed"). Defaults to False.

        Raises:
                ServerError: Unexpected server error

        Returns:
                list: List with the models data that name matches the query
        """
        url = f"{self.base_url}/model/search"

        query = {}

        if name:
            query['name'] = name

        if state:
            query['state'] = state

        if group:
            query['group'] = group

        if only_deployed:
            query['state'] = 'Deployed'

        response = requests.get(url, params=query,
                                                        headers={'Authorization': 'Bearer ' + self.__credentials})
        
        if response.status_code == 200:
            results = response.json()['Results']
            parsed_results = []
            for r in results:
                r['Schema'] = json.loads(r['Schema'])
                parsed_results.append(r)

            return parsed_results
        
        else:
            raise ServerError('Unexpected server error: ', response.text)

    def get_logs(self, model_id, start:Optional[str]=None, end:Optional[str]=None, routine:Optional[str]=None, type:Optional[str]=None):
        url = f"{self.base_url}/model/logs/{model_id}"
        return self._logs(url, self.__credentials, start=start, end=end, routine=routine, type=type)
            
    def __upload_model(self, model_name:str, model_reference:str, source_file:str, 
                            model_file:str, requirements_file:str, schema:Optional[Union[str, dict]]=None, 
                            group:Optional[str]=None, extra_files:Optional[list]=None, env:Optional[str]=None, 
                            python_version:str='3.8', operation:str='Sync', input_type:str=None) -> str:
        """Upload the files to the server

        Args:
                group (Optional[str], optional): Group the model is inserted. If None the server uses 'datarisk' (public group)
                model_name (str): The name of the model, in less than 32 characters
                model_reference (str): The name of the scoring function inside the source file.
                source_file (str): Path of the source file. The file must have a scoring function that accepts two parameters: data (data for the request body of the model) and model_path (absolute path of where the file is located)
                model_file (str): Path of the model pkl file.
                requirements_file (str): Path of the requirements file. The packages versions must be fixed eg: pandas==1.0
                schema (Union[str, dict]): Path to a JSON or XML file with a sample of the input for the entrypoint function. A dict with the sample input can be send as well
                extra_files (Optional[list], optional): A optional list with additional files paths that should be uploaded. If the scoring function refer to this file they will be on the same folder as the source file.
                python_version (str, optional): Python version for the model environment. Avaliable versions are 3.7, 3.8, 3.9, 3.10. Defaults to '3.8'.

        Raises:
                InputError: Some input parameters its invalid

        Returns:
                str: The new model id (hash)
        """
        
        url = f"{self.base_url}/model/upload/{group}"
        
        file_extesions = {'py': 'script.py', 'ipynb': "notebook.ipynb"}
        
     
        upload_data = [
            ("source", (file_extesions[source_file.split('.')[-1]], open(source_file, "r"))),
            ("model", (model_file.split('/')[-1], open(model_file, "rb"))),
            ("requirements", ("requirements.txt", open(requirements_file, "r")))
        ]

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

        if env:
            upload_data.append(("env", (".env", env)))
        
        if extra_files:
            extra_data = [('extra', (c.split('/')[-1], open(c, "r"))) for c in extra_files]
            
            upload_data += extra_data
            
        form_data = {'name': model_name, 'model_reference': model_reference, 'operation': operation, 'input_type': input_type,
                                 'python_version': "Python"+python_version.replace('.', '')}
            
        response = requests.post(url, data=form_data, files=upload_data, headers={'Authorization': 'Bearer ' + self.__credentials})
        
        if response.status_code == 201:
            data = response.json()
            model_id = data["ModelHash"]
            logger.info(f'{data["Message"]} - Hash: "{model_id}"')
            return model_id
        else:
            logger.error('Upload error: ' + response.text)
            raise InputError('Invalid parameters for model creation')

    def __host_model(self, operation:str, model_id:str, group:str) -> None:
        """Builds the model execution environment

        Args:
                operation (str): The model operation type (Sync or Async)
                model_id (str): The uploaded model id (hash)
                group (str): Group the model is inserted. Default is 'datarisk' (public group)

        Raises:
                InputError: Some input parameters its invalid
        """
        
        url = f"{self.base_url}/model/{operation}/host/{group}/{model_id}"
        if operation == 'sync':
            url = url.replace('7070', '7071')

        response = requests.get(url, headers={'Authorization': 'Bearer ' + self.__credentials})
        if response.status_code == 202:
            logger.info(f"Model host in process - Hash: {model_id}")
        else:
            logger.error(response.text)
            raise InputError('Invalid parameters for model creation')

    def create_model(self, model_name:str, model_reference:str, source_file:str, 
                                     model_file:str, requirements_file:str, schema:Optional[Union[str, dict]]=None, 
                                     group:str=None, extra_files:Optional[list]=None, env:Optional[str]=None,
                                     python_version:str='3.8', operation='Sync', input_type:str='json|csv|parquet', 
                                     wait_for_ready:bool=True)-> Union[NeomarilModel, str]:
        """Deploy a new model to Neomaril.

        Args:
                model_name (str): The name of the model, in less than 32 characters
                model_reference (str): The name of the scoring function inside the source file.
                source_file (str): Path of the source file. The file must have a scoring function that accepts two parameters: data (data for the request body of the model) and model_path (absolute path of where the file is located)
                model_file (str): Path of the model pkl file.
                requirements_file (str): Path of the requirements file. The packages versions must be fixed eg: pandas==1.0
                schema (Union[str, dict]): Path to a JSON or XML file with a sample of the input for the entrypoint function. A dict with the sample input can be send as well. Mandatory for Sync models
                group (str): Group the model is inserted. Default to 'datarisk' (public group)
                extra_files (Optional[list], optional): A optional list with additional files paths that should be uploaded. If the scoring function refer to this file they will be on the same folder as the source file.
                source_type (str, optional): The type of the source file. Avaliable values are PythonNotebook (expect a .ipynb file) and PythonScript (expect a .py file). Defaults to 'PythonNotebook'.
                python_version (str, optional): Python version for the model environment. Avaliable versions are 3.7, 3.8, 3.9, 3.10. Defaults to '3.8'.
                wait_for_ready (bool, optional):Wait for model to be ready and returns a NeomarilModel instace with the new model. Defaults to True.

        Raises:
                InputError: Some input parameters its invalid

        Returns:
                Union[NeomarilModel, str]: If wait_for_ready=True runs the deploy process synchronously and returns the new model. If its False, returns nothing after sending all the data to server and runs the deploy asynchronously.
        """
        
        if python_version not in ['3.7', '3.8', '3.9', '3.10']:
            raise InputError('Invalid python version. Avaliable versions are 3.7, 3.8, 3.9, 3.10')
        
        if group:
            group = group.lower().strip().replace(" ", "_").replace(".", "_").replace("-", "_")

            groups = [g["Name"] for g in self.list_groups()]

            if group not in groups:

                raise GroupError('Group dont exist. Create a group first.')

        else:
            group = 'datarisk'
            logger.info("Group not informed, using default 'datarisk' group")
        
        model_id = self.__upload_model(model_name, model_reference, source_file, 
                                                model_file, requirements_file, schema=schema, group=group,
                                                extra_files=extra_files, python_version=python_version,env=env,
                                                operation=operation, input_type=input_type)
                
        self.__host_model(operation.lower(), model_id, group)
        
        if wait_for_ready:
            return self.get_model(model_id, group)
        else:
            return "Model deployment in progress"

    def get_model_execution(self, model_id:str, exec_id:str, group:Optional[str]=None) -> None:
        """Get a execution instace.

        Args:
                exec_id (str): Execution id

        Returns:
                NeomarilExecution: The new execution
        """
        return NeomarilExecution(model_id,'AsyncModel', exec_id, password=self.__credentials, enviroment=self.enviroment, group=group)