Connecting to MLOps
======================

For interacting with MLOps we need to access the clients. 

To interact with the MLOps platform, you will need to access the provided clients.  

**MLOps Codex** offers three primary clients:  

- :py:class:`mlops_codex.training.MLOpsTrainingClient`  
  Used for accessing and managing the training of Machine Learning models.

- :py:class:`mlops_codex.model.MLOpsModelClient`  
  Designed for handling model-related operations, including deployment and monitoring.  

- :py:class:`mlops_codex.datasources.MLOpsDataSourceClient`
  Enables integration and management of data sources for your ML workflows.

- :py:class:`mlops_codex.preprocessing.MLOpsPreprocessingClient`
  Provides tools for managing and automating data preprocessing tasks in Machine Learning workflows.

- :py:class:`mlops_codex.external_monitoring.MLOpsExternalMonitoringClient`
  Enables monitoring of deployed Machine Learning models trained on your own machine.


You need the server URL, an email and a password to access the MLOps. The best way to do it is using a *.env* file with the following env variables

.. code-block:: env

    MLOPS_URL='https://neomaril.datarisk.net'
    MLOPS_USER='email@email.com'
    MLOPS_PASSWORD='password@123'

If you create the `.env` file in the same directory where you are running your code, it will be automatically imported.

.. code:: python

    # Import the client
    from mlops_codex.model import MLOpsModelClient
    from mlops_codex.training import MLOpsTrainingClient
    from mlops_codex.datasources import MLOpsDataSourceClient
    from mlops_codex.preprocessing import MLOpsPreprocessingClient
    from mlops_codex.external_monitoring import MLOpsExternalMonitoringClient

    # Start the client via model client
    model_client = MLOpsModelClient()

    # Start the client via training client
    training_client = MLOpsTrainingClient()

    # Start the client via data source client
    datasource_client = MLOpsDataSourceClient()

    # Start the client via preprocessing client
    preprocessing_client = MLOpsPreprocessingClient()

    # Start the client via external monitoring client
    external_monitoring_client = MLOpsExternalMonitoringClient()


Creating a group
----------------

Groups provide a way to organize training experiments and models that may have different end-users or purposes, enabling the creation of isolated environments for each group. When a group is created, a unique token is generated, which is used to run models and enhance platform security. This token will expire after one year.
Every resource you create in MLOps should belong to a group, making the creation of a group the first step in your workflow.
You can create a group using any of the available clients, simply by providing its name and a description to the group for better clarity and organization.


.. code-block:: python

    # Import the client
    from mlops_codex.training import MLOpsTrainingClient

    training_client = MLOpsTrainingClient()

    training_client.create_group(
        name='nb_demo',
        description='Group for the demo'
    )

    # This token has a 1 year expiration date, to generate a new one use the refresh method

    model_client.refresh_group_token(
        name='nb_demo', # Group name
        force=True # To force creating a new token even if the old is valid
    )

Add your group token to the *.env* file:

.. code-block:: txt

    MLOPS_GROUP_TOKEN='YOUR_GROUP_TOKEN'
