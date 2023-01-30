#!/usr/bin/env python
# coding: utf-8

import io
import re
from typing import Union, Optional
import requests
import json
from neomaril_codex._base import *
from neomaril_codex.model import NeomarilModel
from neomaril_codex.exceptions import *

patt = re.compile(r'(\d+)')

class NeomarilTrainingExecution(NeomarilExecution):
	""" Class to manage trained models
	"""

	def __init__(self, training_id:str, group:str, training_type:str, exec_id:Optional[str]=None, password:str=None, enviroment:str=None) -> None:
		""" Class to manage trained models

		Args:
				training_id (str): Training id (hash) from the experiment you want to acess
				group (str): Group the training is inserted. Default is 'datarisk' (public group)
				exec_id (str): Executiong id for that especific training run
				password (str): Password for authenticating with the client
				enviroment (str): Enviroment of Neomaril you are using. 
	
		Raises:
				TrainingError: When the training can't be acessed in the server
				AuthenticationError: Unvalid credentials
		"""
		super().__init__('Training', exec_id=exec_id, password=password, enviroment=enviroment)
		self.__credentials = password

		self.training_id = training_id
		self.group = group
		self.training_type = training_type

	def __upload_model(self, model_name:str, model_reference:Optional[str]=None, source_file:Optional[str]=None, 
										 schema:Optional[Union[str, dict]]=None, extra_files:Optional[list]=None, 
										 env:Optional[str]=None, operation:str='Sync', input_type:str=None) -> str:
		"""Upload the files to the server

		Args:
				model_name (str): The name of the model, in less than 32 characters
				model_reference (str): The name of the scoring function inside the source file.
				source_file (str): Path of the source file. The file must have a scoring function that accepts two parameters: data (data for the request body of the model) and model_path (absolute path of where the file is located)
				schema (Union[str, dict]): Path to a JSON or XML file with a sample of the input for the entrypoint function. A dict with the sample input can be send as well
				extra_files (Optional[list], optional): A optional list with additional files paths that should be uploaded. If the scoring function refer to this file they will be on the same folder as the source file.

		Raises:
				InputError: Some input parameters its invalid

		Returns:
				str: The new model id (hash)
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
		"""Builds the model execution environment

		Args:
				operation (str): The model operation type (Sync or Async)
				model_id (str): The uploaded model id (hash)

		Raises:
				InputError: Some input parameters its invalid
		"""
		
		url = f"{self.base_url}/model/{operation}/host/{self.group}/{model_id}"
		response = requests.get(url, headers={'Authorization': 'Bearer ' + self.__credentials})
		if response.status_code == 202:
			logger.info(f"Model host in process - Hash: {model_id}")
		else:
			logger.error(response.text)
			raise InputError('Invalid parameters for model creation')

	def promote_model(self, model_name:str, model_reference:Optional[str]=None, source_file:Optional[str]=None, 
										 schema:Optional[Union[str, dict]]=None, extra_files:Optional[list]=None, 
										 env:Optional[str]=None, operation:str='Sync', input_type:str=None)-> NeomarilModel:
			
		model_id = self.__upload_model(model_name, model_reference=model_reference, source_file=source_file, env=env,
																		schema=schema, extra_files=extra_files, operation=operation, 
																		input_type=input_type)
		
		if model_id:
			self.__host_model(operation.lower(), model_id)

			return NeomarilModel(self.__credentials, model_id, group=self.group, test_enviroment=(self.enviroment == 'Staging'))

		
class NeomarilTrainingExperiment(BaseNeomaril):
	""" Class to manage models being trained inside Neomaril

	"""

	def __init__(self, password:str, experiment_name:str, training_id:str, model_type:str, training_type:str, 
							 group:str="datarisk", test_enviroment:bool=True) -> None:
		""" Class to manage models being trained inside Neomaril

		Args:
				password (str): Password for authenticating with the client
				training_id (str): Training id (hash) from the experiment you want to acess
				group (str): Group the training is inserted. Default is 'datarisk' (public group)
				test_enviroment (bool): Flag that choose which enviroment of Neomaril you are using. Test your deployment first before changing to production. Default is True

		Raises:
				TrainingError: When the training can't be acessed in the server
				AuthenticationError: Unvalid credentials
		"""
		super().__init__()
		self.__credentials = password
		self.training_id = training_id
		self.group = group

		if test_enviroment:
			self.enviroment = "Staging"
			self.base_url = self.staging_url

		else:
			raise NotImplementedError
			# self.enviroment = "Production"
			# self.base_url = self.production_url

		try_login(self.__credentials, self.base_url)
			
		self.model_type = model_type
		self.training_type = training_type
		self.experiment_name = experiment_name

	def __repr__(self) -> str:
			return f"""NeomarilTrainingExperiment(name="{self.experiment_name}", 
														group="{self.group}", 
														enviroment="{self.enviroment}"
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
		
		"""Upload the files to the server

		Args:
				model_name (str): The name of the model, in less than 32 characters
				train_data (str): Path of the file with train data.
				
				If training_type is Custom
				training_reference (str): The name of the training function inside the source file.
				source_file (str): Path of the source file. The file must have a training function that accepts one parameter: model_path (absolute path of where the file is located)
				requirements_file (str): Path of the requirements file. The packages versions must be fixed eg: pandas==1.0
				extra_files (Optional[list], optional): A optional list with additional files paths that should be uploaded. If the scoring function refer to this file they will be on the same folder as the source file.
				python_version (str, optional): Python version for the model environment. Avaliable versions are 3.7, 3.8, 3.9, 3.10. Defaults to '3.8'.

				If training_type is AutoML
				conf_dict (Union[str, dict]): Path to a JSON file with a the AutoML configuration. A dict can be send as well

		Raises:
				InputError: Some input parameters its invalid

		Returns:
				str: The new model id (hash)
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
		"""Builds the model execution environment

		Args:
				exec_id (str): The uploaded training execution id (hash)

		Raises:
				InputError: Some input parameters its invalid
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
												extra_files:Optional[list]=None) -> Union[dict, NeomarilExecution]:
		"""Runs a prediction from the current model.

		Args:
				model_name (str): The name of the model, in less than 32 characters
				train_data (str): Path of the file with train data.
				
				If training_type is Custom
				training_reference (str): The name of the training function inside the source file.
				source_file (str): Path of the source file. The file must have a training function that accepts one parameter: model_path (absolute path of where the file is located)
				requirements_file (str): Path of the requirements file. The packages versions must be fixed eg: pandas==1.0
				extra_files (Optional[list], optional): A optional list with additional files paths that should be uploaded. If the scoring function refer to this file they will be on the same folder as the source file.
				python_version (str, optional): Python version for the model environment. Avaliable versions are 3.7, 3.8, 3.9, 3.10. Defaults to '3.8'.

				If training_type is AutoML
				conf_dict (Union[str, dict]): Path to a JSON file with a the AutoML configuration. A dict can be send as well

		Raises:
				ModelError: Model is not available

		Returns:
				Union[dict, NeomarilExecution]: The return of the scoring function in the source file for Sync models or the execution class for Async models.
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
			return NeomarilTrainingExecution(self.training_id, self.group, self.training_type, exec_id, password=self.__credentials, enviroment=self.enviroment)

	def __call__(self, data: dict) -> dict:
			return self.predict(data)

	def get_training_execution(self, exec_id:str) -> None:
		"""Get a execution instace.

		Args:
				exec_id (str): Execution id

		Returns:
				NeomarilExecution: The new execution
		"""
		return NeomarilTrainingExecution(self.training_id, self.group, self.training_type, exec_id, password=self.__credentials, enviroment=self.enviroment)

class NeomarilTrainingClient(BaseNeomarilClient):
	"""Client for acessing Neomaril and manage models

	"""
	def __init__(self, password:str, test_enviroment:bool=True) -> None:
		"""Client for acessing Neomaril and manage models

		Args:
				password (str): Password for authenticating with the client
				test_enviroment (bool): Flag that choose which enviroment of Neomaril you are using. Test your deployment first before changing to production. Default is True

		Raises:
				AuthenticationError: Unvalid credentials
				ServerError: Server unavailable
		"""
		super().__init__(password, test_enviroment=test_enviroment)
		self.__credentials = password
			
	def __repr__(self) -> str:
			return f'NeomarilTrainingClient(enviroment="{self.enviroment}", version="{self.client_version}")'
		
	def __str__(self):
		return f"NEOMARIL {self.enviroment} Training client:{self.client_version}"
		
	
	# def get_training(self, training_id:str, group:str="datarisk") -> NeomarilTrainingExperiment:
	#   """Acess a model using its id

	#   Args:
	#       training_id (str): Training id (hash) that needs to be acessed
	#       group (str): Group the model is inserted. Default is 'datarisk' (public group)

	#   Raises:
	#       TrainingError: Model unavailable
	#       ServerError: Unknown return from server

	#   Returns:
	#       NeomarilTrainingExperiment: A NeomarilTrainingExperiment instance with the training hash from `training_id`
	#   """

	#   return NeomarilTrainingExperiment(self.__credentials, training_id, group=group)
	

	def create_training_experiment(self, experiment_name:str, model_type:str, training_type:str, group:str='datarisk')-> NeomarilTrainingExperiment:
		"""Create a new training experiment on Neomaril.

		Args:
				experiment_name (str): The name of the experiment, in less than 32 characters
				model_type (str): The name of the scoring function inside the source file.
				training_type (str): Path of the source file. The file must have a scoring function that accepts two parameters: data (data for the request body of the model) and model_path (absolute path of where the file is located)
				group (str): Group the model is inserted. Default to 'datarisk' (public group)

		Raises:
				InputError: Some input parameters its invalid

		Returns:
				NeomarilTrainingExperiment: 
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

		
		return NeomarilTrainingExperiment(self.__credentials, experiment_name, training_id, model_type, training_type, group=group)     