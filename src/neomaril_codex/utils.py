import requests
from neomaril_codex.exceptions import *

def parse_url(url):
    if url.endswith('/'):
        url = url[:-1]

    if not url.endswith('/api'):
        url = (url+'/api')
    return url

def try_login(password:str, base_url:str) -> bool:

    response = requests.get(f"{base_url}/health", headers={'Authorization': 'Bearer ' + password})

    server_status = response.status_code

    if server_status == 200:
      return response.json()['Version']

    elif server_status == 401:
      raise AuthenticationError('Invalid credentials.')

    elif server_status >= 500:
      raise ServerError('Neomaril server unavailable at the moment.')