import requests, os
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger
from dotenv import load_dotenv

from neomaril_codex.exceptions import *

def try_login(password:str, base_url:str) -> bool:

    response = requests.get(f"{base_url}/health", headers={'Authorization': 'Bearer ' + password})

    server_status = response.status_code

    if server_status == 200:
      return response.json()['Version']

    elif server_status == 401:
      raise AuthenticationError('Invalid credentials.')

    elif server_status >= 500:
      raise ServerError('Neomaril server unavailable at the moment.')

class BaseNeomaril:
    """Base class for others Neomaril related classes.
    """

    def __init__(self) -> None:
        self._production_url = "http://neomaril.datarisk.net/api"
        self._staging_url = "http://neomaril.staging.datarisk.net/api"
        self._dev_url = "http://localhost:7070/api"


    def _logs(self, url, creds, start:Optional[str]=None, end:Optional[str]=None, routine:Optional[str]=None, type:Optional[str]=None):
       
        if not start and not end:
            end = datetime.today().strftime("%d-%m-%Y")
            start = (datetime.today() - timedelta(days=6)).strftime("%d-%m-%Y")

        if not start and end:
            start = (datetime.strptime(end, "%d-%m-%Y") - timedelta(days=6)).strftime("%d-%m-%Y")

        if start and not end:
            end = (datetime.strptime(start, "%d-%m-%Y") + timedelta(days=6)).strftime("%d-%m-%Y")

        query = {'start': start, 'end': end}

        if routine:
            assert routine in ['Run', 'Host']
            query['routine'] = routine

        if type:
            assert type in ['Ok', 'Error', 'Debug', 'Warning']
            query['type'] = type

        response = requests.get(url, params=query,
                            headers={'Authorization': 'Bearer ' + creds})
    
        if response.status_code == 200: 
            return response.json()
        else:
            raise ServerError('Unexpected server error: ', response.text)


class BaseNeomarilClient(BaseNeomaril):
	"""Base class for others client related classes.
	"""
	def __init__(self, password:str='', enviroment:str='staging') -> None:
		super().__init__()
		load_dotenv()

		self.__credentials = os.getenv('NEOMARIL_TOKEN') if os.getenv('NEOMARIL_TOKEN') else password
		self.enviroment = os.getenv('NEOMARIL_ENVIROMENT') if os.getenv('NEOMARIL_ENVIROMENT') else enviroment

		if self.enviroment == 'dev':
				self.base_url = self._dev_url

		elif self.enviroment == 'staging':
				self.base_url = self._staging_url
				logger.info("You are using the test enviroment that will have the data cleaned from time to time. If your model is ready to use change the enviroment to Production")

		elif self.enviroment == 'production':
				raise NotImplementedError
				# self.enviroment = "Production"
				# self.base_url = self._production_url
				# logger.info("You are using the production enviroment, please use the test enviroment if you are still developing the model.")

		self.client_version = try_login(self.__credentials, self.base_url)
		logger.info(f"Successfully connected to Neomaril")


	def list_groups(self) -> list:
		"""List existing groups

		Raises:
				ServerError: Unexpected server error

		Returns:
				list: List with the groups that exists in the database
		"""

		url = f"{self.base_url}/groups"
		response = requests.get(url, headers={'Authorization': 'Bearer ' + self.__credentials})

		if response.status_code == 200:
				results = response.json()['Results']

				return results
		else:
				raise ServerError('Unexpected server error: ', response.text)


	def create_group(self, name:str, description:str) -> bool:
		"""Create a group for multiple models of the same final client

		Args:
				name (str): Name of the group. Must be 32 characters long and with no special characters (some parsing will be made).
				description (str): Short description of the group.

		Raises:
				ServerError: Unexpected server error

		Returns:
				bool: Returns True if the group was successfully created and False if not.
		"""
		data = {"name": name, "description": description}

		url = f"{self.base_url}/groups"
		response = requests.post(url, data=data,
														 headers={'Authorization': 'Bearer ' + self.__credentials})

		if response.status_code == 201:
				logger.info(response.json()['Message'])
				return True
		elif response.status_code == 400:
				logger.error("Group already exist, nothing was changed.")
				return False
		else:
				raise ServerError('Unexpected server error: ', response.text)

	def refresh_group_token(self, name:str, force:bool=False) -> bool:
		"""Create a group for multiple models of the same final client

		Args:
				name (str): Name of the group to have the token refreshed.
				force (str): Force token expiration even if its still valid (this can make multiple models integrations stop working, so use with care).

		Raises:
				ServerError: Unexpected server error

		Returns:
				bool: Returns True if the group was successfully created and False if not.
		"""

		url = f"{self.base_url}/refresh/{name}"
		response = requests.get(url, params={'force': str(force).lower()},
														 headers={'Authorization': 'Bearer ' + self.__credentials})

		if response.status_code == 201:
				return response.json()['Message']
		else:
				raise ServerError('Unexpected server error: ', response.text)


class NeomarilExecution(BaseNeomaril):
	"""Base class for Neomaril async executions.
	"""

	def __init__(self, parent_id:str, exec_type:str, group:Optional[str]=None, exec_id:Optional[str]=None, password:str=None, enviroment:str=None) -> None:
		super().__init__()
		self.enviroment = enviroment
		self.exec_type = exec_type
		self.exec_id = exec_id
		self.status = 'Requested'
		self.__credentials = password

		if enviroment == "Staging":
				self.base_url = self._staging_url
				self.mlflow_url = 'https://mlflow.staging.datarisk.net/'
		else:
				self.base_url = self._production_url
				self.mlflow_url = 'https://mlflow.datarisk.net/'

		try_login(self.__credentials, self.base_url)

		if exec_type == 'AsyncModel':
				self.__url_path = 'model/async'
				del self.mlflow_url
		elif exec_type == 'Training':
				self.__url_path = 'training'
		else:
				raise InputError(f"Invalid execution type '{exec_type}'. Valid options are 'AsyncModel' and 'Training'")

		url = f"{self.base_url}/{self.__url_path.replace('/async', '')}/describe/{group}/{parent_id}/{exec_id}"
		response = requests.get(url, headers={'Authorization': 'Bearer ' + self.__credentials})
		
		if response.status_code == 404:
				raise ModelError(f'Execution "{exec_id}" not found.')
		
		elif response.status_code >= 500:
				raise ModelError(f'Unable to retrive execution "{exec_id}"')
		

		self.execution_data = response.json()['Description']

		self.status = self.execution_data['ExecutionState']

	def __repr__(self) -> str:
		return f"""Neomaril{self.exec_type}Execution(exec_id="{self.exec_id}", status="{self.status}")"""

	def __str__(self):
		return f'NEOMARIL {self.exec_type }Execution :{self.exec_id} (Status: {self.status})"'

	def get_status(self) -> dict:
		"""Gets the status of the execution with the informed id

		Raises:
				ExecutionError: Execution unavailable

		Returns:
				dict: Returns the execution status.
		"""

		url = f"{self.base_url}/{self.__url_path}/status/{self.exec_id}"
		response = requests.get(url, headers={'Authorization': 'Bearer ' + self.__credentials})
		if response.status_code not in [200, 410]:
				logger.error(response.text)
				raise ExecutionError(f'Execution "{self.exec_id}" unavailable')

		result = response.json()

		self.status = result['Status']
		self.execution_data['Status'] = result['Status']

		if self.status == 'Succeeded':
				if self.exec_type == 'Training':
						logger.info(f'You can check the run info in {self.mlflow_url} ')

		return result

	def download_result(self, path:Optional[str]='./') -> dict:
		"""Gets the output of the execution with the informed id

		Raises:
				ExecutionError: Execution unavailable

		Returns:
				dict: Returns the execution status.
		"""
		if self.status in ['Running', 'Requested']:
			self.status = self.get_status()['Status']

		if self.status == 'Succeeded':
				url = f"{self.base_url}/{self.__url_path}/result/{self.exec_id}"
				response = requests.get(url, headers={'Authorization': 'Bearer ' + self.__credentials})
				if response.status_code not in [200, 410]:
						logger.error(response.text)
						raise ExecutionError(f'Execution "{self.exec_id}" unavailable')

				filename = f'output_{self.exec_id}.zip'
				if not path.endswith('/'):
						filename = '/'+filename

				with open(path+filename, 'wb') as f:
						f.write(response.content)

				logger.info(f'Output saved in {path+filename}')
		elif self.status == 'Failed':
				raise ExecutionError("Execution failed")
		else:
				logger.info(f'Execution not ready. Status is {self.status}')