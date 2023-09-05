import requests
from neomaril_codex.exceptions import *
from msal import PublicClientApplication

CLIENT_ID = '1432430b-d7b5-4081-b46b-7ca8ddffda26'
AUTHORITY = 'https://datariskb2c.b2clogin.com/datariskb2c.onmicrosoft.com/B2C_1_NEOMARIL_API_SIGNIN'

app = PublicClientApplication(CLIENT_ID, authority=AUTHORITY)

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
      token = app.acquire_token_by_username_password(login, password, [CLIENT_ID])
      if 'access_token' in token.keys():
        return response.json()['Version']
      else:
        raise AuthenticationError(token.get('error_description', 'Invalid credentials.'))

    elif server_status == 401:
      raise AuthenticationError('Invalid credentials.')

    elif server_status >= 500:
      raise ServerError('Neomaril server unavailable at the moment.')
    

def refresh_token(login:str, password:str):
  if app.get_accounts():
    token = app.acquire_token_silent([CLIENT_ID], account=[acc for acc in app.get_accounts() if acc['username'] == login][0])
  else:
    token = app.acquire_token_by_username_password(login, password, [CLIENT_ID])
  
  if 'access_token' in token.keys():
    return token['access_token']
  else:
    raise AuthenticationError(token.get('error_description', 'Invalid credentials.'))
