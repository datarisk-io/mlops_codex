#!/usr/bin/env python
# coding: utf-8

import io
from typing import Union, Optional
from time import sleep
import requests
import json
from neomaril_codex.logging import Logger
from neomaril_codex._exceptions import *

class BaseNeomaril(object):
  """Base class for others Neomaril related classes.
  """

  def __init__(self) -> None:
    self.base_url = "http://neomaril.datarisk.net/api"
    self.log = Logger()
    
class NeomarilModel(BaseNeomaril):
  """ Class to manage Models deployed inside Neomaril

  """

  def __init__(self, username:str, password:str, model_id:str) -> None:
    """Class to manage Models deployed inside Neomaril

    Args:
        username (str): User for authenticating with the client
        password (str): Password for authenticating with the client
        model_id (str): Model id (hash) from the model you want to acess

    Raises:
        ModelError: When the model can't be acessed in the server
        AuthenticationError: Unvalid credentials
    """
    super().__init__()
    self.__credentials = (username, password)
    self.model_id = model_id
    
    url = f"{self.base_url}/describe/{self.model_id}"
    response = requests.get(url, auth=self.__credentials)
    
    if response.status_code != 200:
      raise ModelError(f'Unable to retrive model "{model_id}"')
      
    elif response.status_code == 401:
      raise AuthenticationError('Invalid credentials.')
    
    self.model_data = response.json()['Description']
    self.name = self.model_data['Name']
    self.status = self.model_data['Status']
    self.schema = self.model_data['Schema']
    self.__model_ready = True

  def __repr__(self) -> str:
      return f"""NeomarilModel(name="{self.name}", status="{self.status}",
                               model_id="{self.model_id}",
                               schema={str(self.schema)}
                               )"""

  def __str__(self):
    return f'NEOMARIL model "{self.model_id}"'
  
  def delete(self):
    """Deletes the current model.
    IMPORTANT! For now this is irreversible, if you want to use the model again you will need to upload again (and it will have a new ID).

    Raises:
        ServerError: _description_

    Returns:
        _type_: _description_
    """
    if self.__model_ready:
      req = requests.delete(f"{self.base_url}/{self.model_id}", auth=self.__credentials)
      
      if req.status_code == 200:
        response = requests.get(f"{self.base_url}/describe/{self.model_id}", 
                                auth=self.__credentials)
        
        self.model_data = response.json()['Description']
        self.status = self.model_data['Status']
        self.__model_ready = False
      
        return req.json()
      else:
        raise ServerError('Model deleting failed')
      
    else:
      return 'Model is '+self.status

  def predict(self, data:dict) -> dict:
    """Runs a prediction from the current model.

    Args:
        data (dict): The same data that is used in the source file. The keys that are needed inside this dict are the ones in the `schema` atribute.

    Raises:
        ModelError: Model is not available

    Returns:
        dict: The return of the scoring function in the source file.
    """
    if self.__model_ready:
      url = f"{self.base_url}/run/{self.model_id}"

      model_input = {
          "Input": data
      }

      req = requests.post(url, data=json.dumps(model_input), auth=self.__credentials)

      return req.json()
    
    else:
      raise ModelError('Model is not available to predictions')

  def __call__(self, data: dict) -> dict:
      return self.predict(data)

class NeomarilClient(BaseNeomaril):
  """Client for acessing Neomaril and manage models

  """
  def __init__(self, username:str, password:str) -> None:
    """Client for acessing Neomaril and manage models

    Args:
        username (str): User for authenticating with the client
        password (str): Password for authenticating with the client

    Raises:
        AuthenticationError: Unvalid credentials
        ServerError: Server unavailable
    """
    super().__init__()
    self.__credentials = (username, password)

    response = requests.get(f"{self.base_url}/health", auth=self.__credentials)
    
    server_status = response.status_code
    
    if server_status == 418:
      self.client_version = response.json()['Version']
      self.log.info(f"User '{username}' successfully connected to Neomaril")
      
    elif server_status == 401:
      raise AuthenticationError('Invalid credentials.')
    
    elif server_status >= 500:
      raise ServerError('Neomaril server unavailable at the moment.')
    
  def __repr__(self) -> str:
      return f'NeomarilClient(username="{self.__credentials[0]}", version="{self.client_version}")'
    
  def __str__(self):
    return "NEOMARIL client"
    
  def __get_model_status(self, model_id:str) -> dict:
    """Gets the status of the model with the hash equal to `model_id`

    Args:
        model_id (str): Model id (hash) from the model being searched

    Raises:
        ModelError: Model unavailable

    Returns:
        dict: Returns the model status and a message if the status is 'Failed'.
    """

    url = f"{self.base_url}/status/{model_id}"
    response = requests.get(url, auth=self.__credentials)
    if response.status_code not in [200, 410]:
      raise ModelError(f'Model "{model_id}" not found')
    
    return response.json()
  
  def get_model(self, model_id:str, wait_for_ready:bool=True) -> NeomarilModel:
    """Acess a model using its id

    Args:
        model_id (str): Model id (hash) that needs to be acessed
        wait_for_ready (bool, optional): If the model is being deployed, wait for it to be ready instead of failing the request. Defaults to True.

    Raises:
        ModelError: Model unavailable
        ServerError: Unknown return from server

    Returns:
        NeomarilModel: A NeomarilModel instance with the model hash from `model_id`
    """
    response = self.__get_model_status(model_id)
    
    status = response['Status']
    
    if status == 'Building':
      if wait_for_ready:
        print('Wating for deploy to be ready.', end='')
        while status == 'Building':
          status = self.__get_model_status(model_id)['Status']
          print('.', end='', flush=True)
          sleep(10)
      else:
        raise ModelError(f'Model "{model_id}" not ready yet')
      
    if status in ['Disabled', 'Ready']:
      raise ModelError(f'Model "{model_id}" unavailable (disabled or deploy process is incomplete)')
    elif status == 'Failed':
      self.log.error(str(response['Message']))
      raise ModelError(f'Model "{model_id}" deploy failed, so model is unavailable.')
    elif status == 'Deployed': 
      self.log.info(f'Model {model_id} its deployed. Fetching model.')
      return NeomarilModel(*self.__credentials, model_id)
    else:
      raise ServerError('Unknown model status: ',status)
  
  def search_models(self, query:str, only_deployed:bool=False) -> list:
    """Search for models using the name of the model

    Args:
        query (str): Text that its expected to be on the model name. It runs similar to a LIKE query on SQL.
        only_deployed (bool, optional): If its True, filter only models ready to be used (status == "Deployed"). Defaults to False.

    Raises:
        ServerError: Unexpected server error

    Returns:
        list: List with the models data that name matches the query
    """
    url = f"{self.base_url}/search/{query}"
    response = requests.get(url, auth=self.__credentials)
    
    if response.status_code == 200:
      results = response.json()['Results']
      
      if only_deployed:
        return [r for r in results if r['Status'] == 'Deployed']
      else:
        return results
    else:
      raise ServerError('Unexpected server error: ', response.text)
      
  def __upload_model(self, model_name:str, model_reference:str, source_file:str, 
                     model_file:str, requirements_file:str, schema:Union[str, dict], 
                     extra_files:Optional[list]=None, source_type:str='PythonNotebook') -> str:
    """Upload the files to the server

    Args:
        model_name (str): The name of the model, in less than 32 characters
        model_reference (str): The name of the scoring function inside the source file.
        source_file (str): Path of the source file. The file must have a scoring function that accepts two parameters: data (data for the request body of the model) and model_path (absolute path of where the file is located)
        model_file (str): Path of the model pkl file.
        requirements_file (str): Path of the requirements file. The packages versions must be fixed eg: pandas==1.0
        schema (Union[str, dict]): Path to a JSON or XML file with a sample of the input for the entrypoint function. A dict with the sample input can be send as well
        extra_files (Optional[list], optional): A optional list with additional files paths that should be uploaded. If the scoring function refer to this file they will be on the same folder as the source file.
        source_type (str, optional): The type of the source file. Avaliable values are PythonNotebook (expect a .ipynb file) and PythonScript (expect a .py file). Defaults to 'PythonNotebook'.

    Raises:
        InputError: Some input parameters its invalid

    Returns:
        str: The new model id (hash)
    """
    
    url = f"{self.base_url}/upload"
    
    file_extesions = {'py': 'script.py', 'ipynb': "notebook.ipynb"}
    
    if isinstance(schema, str):
      schema_file = open(schema, "r")
    elif isinstance(schema, dict):
      schema_file = io.StringIO()
      json.dump(schema, schema_file).seek(0)
    
    upload_data = {
      "source": (file_extesions[source_file.split('.')[-1]], open(source_file, "r")),
      "model": (model_file.split('/')[-1], open(model_file, "rb")),
      "requirements": ("requirements.txt", open(requirements_file, "r")),
      "schema": ("schema.json", schema_file)
    }
    
    if extra_files:
      extra_data = {'extra'+str(i): (c.split('/')[-1], open(c, "r")) 
                    for i,c in enumerate(extra_files)}
      
      upload_data = {**upload_data, **extra_data}
      
    form_data = {'name': model_name, 'model_reference': model_reference,
                 'type': source_type}
    
    response = requests.put(url, data=form_data, files=upload_data, auth=self.__credentials)
    
    if response.status_code == 201:
      data = response.json()
      model_id = data["ModelHash"]
      self.log.info(f'{data["Message"]} - Hash: "{model_id}"')
      return model_id
    else:
      self.log.error('Upload error: ', response.text)
      raise InputError('Invalid parameters for model creation')

  def __host_model(self, model_id:str, python_version:str='3.8') -> None:
    """Builds the model execution environment

    Args:
        model_id (str): The uploaded model id (hash)
        python_version (str, optional): Python version for the model environment. Avaliable versions are 3.7, 3.8, 3.9, 3.10. Defaults to '3.8'.

    Raises:
        InputError: Some input parameters its invalid
    """
    
    url = f"{self.base_url}/host/python{python_version.replace('.', '')}/{model_id}"
    response = requests.get(url, auth=self.__credentials)
    if response.status_code == 202:
      self.log.info(f"Model host in process - Hash: {model_id}")
    else:
      raise InputError('Invalid parameters for model creation')

  def create_model(self, model_name:str, model_reference:str, source_file:str, 
                   model_file:str, requirements_file:str, schema:Union[str, dict], 
                   extra_files:Optional[list]=None, source_type:str='PythonNotebook', 
                   python_version:str='3.8', wait_for_ready:bool=True)-> Union[NeomarilModel, str]:
    """Deploy a new model to Neomaril.

    Args:
        model_name (str): The name of the model, in less than 32 characters
        model_reference (str): The name of the scoring function inside the source file.
        source_file (str): Path of the source file. The file must have a scoring function that accepts two parameters: data (data for the request body of the model) and model_path (absolute path of where the file is located)
        model_file (str): Path of the model pkl file.
        requirements_file (str): Path of the requirements file. The packages versions must be fixed eg: pandas==1.0
        schema (Union[str, dict]): Path to a JSON or XML file with a sample of the input for the entrypoint function. A dict with the sample input can be send as well
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
    
    model_id = self.__upload_model(model_name, model_reference, source_file, 
                     model_file, requirements_file, schema=schema, 
                     extra_files=extra_files, source_type=source_type)
    
    self.__host_model(model_id, python_version)
    
    if wait_for_ready:
      return self.get_model(model_id)
    else:
      return "Model deployment in progress"