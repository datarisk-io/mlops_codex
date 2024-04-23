Connecting to Neomaril
======================

For interacting with Neomaril we need to access the clients. 

We have 3 of them: :py:class:`neomaril_codex.training.NeomarilTrainingClient`, :py:class:`neomaril_codex.model.NeomarilModelClient` and :py:class:`neomaril_codex.datasource.NeomarilDataSourceClient`.

You need the server URL, an email and a password to access the Neomaril. The best way to do it is using a *.env* file with the following env variables

.. code::

    NEOMARIL_URL='https://neomaril.datarisk.net'
    NEOMARIL_USER='email@email.com'
    NEOMARIL_PASSWORD='password@123'

If you create this file in the same directory your are running your code we will import it automatically

.. code:: python

    # Import the client
    from neomaril_codex.model import NeomarilModelClient
    from neomaril_codex.training import NeomarilTrainingClient
    from neomaril_codex.datasources import NeomarilDataSourceClient

    # Start the client via model client
    model_client = NeomarilModelClient()
    #>>> 2023-10-25 08:37:50.465 | INFO     | neomaril_codex.model:__init__:722 - Loading .env
    #>>> 2023-10-25 08:37:50.466 | INFO     | neomaril_codex.base:__init__:90 - Loading .env
    #>>> 2023-10-25 08:37:52.698 | INFO     | neomaril_codex.base:__init__:102 - Successfully connected to Neomaril

    # Start the client via training client
    training_client = NeomarilTrainingClient()
    #>>> 2023-05-24 10:58:24.855 | INFO     | neomaril_codex.base:__init__:87 - Loading .env
    #>>> 2023-05-24 10:58:25.028 | INFO     | neomaril_codex.base:__init__:99 - Successfully connected to Neomaril
    #>>> 2023-05-24 10:58:25.028 | INFO     | neomaril_codex.base:__init__:102 - Successfully connected to Neomaril

    # Start the client via data source client
    datasource_client = NeomarilDataSourceClient()
    #>>> 2024-03-20 19:19:35.385 | INFO     | neomaril_codex.base:__init__:20 - Loading .env
    #>>> 2024-03-20 19:19:37.219 | INFO     | neomaril_codex.base:__init__:30 - Successfully connected to Neomaril

Creating a group
----------------

Groups are a way to separate training experiments and models that might have different end-users. 
We use it to organize the file system and network in a way that we can create a isolated process for a group. When a group is created a unique token is created to it, this is used to run the models and also increase the security of the platform.

Every resource you create in Neomaril should be in a group, so creating one should be the first thing you do.

To create a group you can use any client, we just need its name. But we also could add description to it.

.. code:: python

   # Import the client
    from neomaril_codex.training import NeomarilTrainingClient

    model_client = NeomarilModelClient()

    model_client.create_group(
        name='nb_demo', # Group name
        description='Group for the demo' # A small description
    )

    #>>> 2023-05-24 10:58:25.634 | INFO     | neomaril_codex.base:create_group:155 - Group 'nb_demo' inserted. Use the following token for scoring: 'f376c18092314246a432a2882c3cc8fd'. Carefully save it as we won't show it again.' 

    # We create a separate group token to be used in model predictions, so it can be shared with the clients
    # This token has a 1 year expiration date, to generate a new one use the refresh method

    model_client.refresh_group_token(
        name='nb_demo', # Group name
        force=True # To force creating a new token even if the old is valid
    )

Add your group token to the *.env* file:

.. code::
    NEOMARIL_GROUP_TOKEN='YOUR_GROUP_TOKEN'
