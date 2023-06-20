import requests
from neomaril_codex.exceptions import *
from msal import PublicClientApplication

app = PublicClientApplication('676f0aa7-0eba-4ac6-ae3b-d93a5e487803', 
                               authority='https://datariskauth.b2clogin.com/datariskauth.onmicrosoft.com/B2C_1_NI')

def parse_url(url):
    if url.endswith('/'):
        url = url[:-1]

    if not url.endswith('/api'):
        url = (url+'/api')
    return url

def try_login(login:str, password:str, base_url:str) -> bool:

    response = requests.get(f"{base_url}/health")

    server_status = response.status_code

    if server_status == 200:
      token = app.acquire_token_by_username_password(login, password, ['676f0aa7-0eba-4ac6-ae3b-d93a5e487803'])
      if 'access_token' in token.keys():
        return response.json()['Version']
      else:
        raise AuthenticationError(token.get('error_description', 'Invalid credentials.'))

    elif server_status == 401:
      raise AuthenticationError('Invalid credentials.')

    elif server_status >= 500:
      raise ServerError('Neomaril server unavailable at the moment.')
    

def refresh_token(login:str, password:str):
  token = app.acquire_token_silent(['676f0aa7-0eba-4ac6-ae3b-d93a5e487803'], account=app.get_accounts()[0])
  if 'access_token' in token.keys():
    return token['access_token']
  else:
    raise AuthenticationError(token.get('error_description', 'Invalid credentials.'))