Deploying to production
=======================

When deploying a model using MLOps Codex, an API is created to facilitate the integration of your model with other services. Additionally, MLOps Codex allows you to execute your model remotely within a Python application.


Preparing to production
------------------------

The first requirement is the scoring script. Similar to the training process, this script needs an entry point function. The parameters and return value of this function will depend on the specific operation of the model.


**Sync model:** This is the "real time" model. The model will expect a JSON and return a JSON after a few seconds.
The entrypoint function should look like this:

.. code-block:: python

    def score(data:str, base_path:str): # It is the name of the function (score) that must be passed in the 'model_reference' field

        # Environment variables loaded from a user-supplied file in the 'env' field
        # my_var = os.getenv('MY_VAR')
        # if my_var is None:
        #   raise Exception("Could not find `env` variable")

        ## MLOps environment variable with model file name
        # with open(base_path+os.getenv('modelFileName'), 'rb') as f:

        # Loads the already trained model to be run based on the model file passed as a parameter
        with open(base_path+"/model.pkl", 'rb') as f:
            model = load(f)

        # Build a DataFrame with the input data. The data arrives as a JSON string so we have to transform it into a dictionary
        df = pd.DataFrame(data=json.loads(data), index=[0])
        
        # Returns the results of running the model as a dictionary
        # It's important to note that in this case, as we're converting to JSON, we can't use numpy types, so we convert to pure int and float.
        return {"pred": int(model.predict(df)), "proba": float(model.predict_proba(df)[0,1])}

The first parameter is the JSON data to be sent to the model, which is provided as a JSON string and should be parsed as needed.

The second parameter is the path, which can be used to access the model files and any other files you upload, similar to the training process.

The function should return a dictionary that can be converted to JSON or a valid JSON string.

Please note that certain data types, such as numpy `int64` and `float64`, cannot typically be parsed to JSON. Therefore, your code should address this before returning the response to MLOps.

**Async model:** This is for batch scoring. We send files with usually a lot of records at once. Since this might take a while depending on the file size, we run this asynchronously.

The entrypoint function should look like this:

.. code-block:: python

    def score(data_path:str, model_path:str):## Environment variables loaded from a file supplied by the user in the 'env' field

        # my_var = os.getenv('MY_VAR')
        # if my_var is None:
        #    raise Exception("Could not find `env` variable")

        ## Environment variable loaded from MLOps with model file name
        # with open(base_path+os.getenv('modelFileName'), 'rb') as f:

        # Loads the already trained model to be run based on the model file passed as a parameter
        with open(model_path+"/model.pkl", 'rb') as f:
            model = load(f)

        ## Environment variable loaded from MLOps with database name
        # X = pd.read_csv(data_path+'/'+os.getenv('inputFileName'))

        # Loads the input base data from the file into a DataFrame
        X = pd.read_csv(data_path+"/dados.csv")

        df = X.copy()   # Creates a copy of the DataFrame with the input data

        df['proba'] = model.predict_proba(X)[:,1]   # Calculates the prediction for each entry in the data table
        df['pred'] = model.predict(X)               # Calculates the probability of each entry in the data table

        # Create the path with the name of the output file, in this case 'output.csv'. It is important that this file is saved in the same path as the data that was sent.
        output = data_path+'/output.csv'

        # Transform the DataFrame, with the prediction and probability, to csv by placing it in the output path file
        df.to_csv(output, index=False)

        # Returns the path to the file with the results of the model run
        return output

The first parameter now serves as a data path. We have distinct path parameters because each asynchronous model execution is stored in a different location. The files uploaded during model deployment remain consistent each time.

To maintain a more dynamic code structure without enforcing a specific file name pattern, you can utilize the `inputFileName` environment variable, which corresponds to the filename uploaded for that execution.

You must save the result in the same path where the input file was located. The function should return this full path.

Deploying your model
--------------------

With all files ready we can deploy the model in two ways.

- Using the :py:meth:`mlops_codex.training.MLOpsTrainingExecution.promote_model` to promote a succeeded training execution.

.. code-block:: python

    # Promoting a custom training execution
    model = custom_run.promote_model(
        model_name='Teste notebook promoted custom', # model_name
        model_reference='score', # name of the scoring function
        source_file=PATH+'app.py', # Path of the source file
        schema=PATH+'schema.json', # Path of the schema file, but it could be a dict (only required for Sync models)
        # env=PATH+'.env'  #  File for env variables (this will be encrypted in the server)
        # extra_files=[PATH+'utils.py'], # List with extra files paths that should be uploaded along (they will be all in the same folder)
        operation="Sync" # Can be Sync or Async
    )

    # Promoting an AutoML training execution
    model = automl_run.promote_model(
        model_name='Teste notebook promoted autoML', # model_name
        operation="Async" # Can be Sync or Async
    )



- Using the :py:meth:`mlops_codex.model.MLOpsModelClient.create_model` to deploy a model trained outside MLOps

.. code-block:: python
    
    # Deploying a new model
    model = client.create_model(
        model_name='Teste notebook Sync', # model_name
        model_reference='score', # name of the scoring function
        source_file=PATH+'app.py', # Path of the source file
        model_file=PATH+'model.pkl', # Path of the model pkl file, 
        requirements_file=PATH+'requirements.txt', # Path of the requirements file, 
        schema=PATH+'schema.json', # Path of the schema file, but it could be a dict (only required for Sync models)
        # env=PATH+'.env'  #  File for env variables (this will be encrypted in the server)
        # extra_files=[PATH+'utils.py'], # List with extra files paths that should be uploaded along (they will be all in the same folder)
        python_version='3.9', # Can be 3.8 to 3.10
        operation="Sync", # Can be Sync or Async
        group='datarisk' # Model group (create one using the client)
    )


Deploying a pre-trained model in MLOps requires minimal information, as AutoML models need only two parameters.

If the deployment succeeds you can start using your model.

These methods return an instance of :py:class:`mlops_codex.model.MLOpsModel`. You can utilize the wait_for_ready parameter during deployment or invoke the :py:meth:`mlops_codex.model.MLOpsModel.wait_ready` method to ensure the :py:class:`mlops_codex.model.MLOpsModel` instance is ready for use. We will install the necessary model dependencies (if you are promoting a training, we will use the same dependencies as the training execution) and conduct some tests. For synchronous models, a sample JSON of the expected API schema is required.

If the deployment is successful, you can begin using your model.

Using your model
----------------

We can use the same :py:class:`mlops_codex.model.MLOpsModel` instance to call the model.

.. code-block:: python

    # For sync models
    sync_model.predict(data={'key': 'value'})

    # For async models
    execution = async_model.predict(data=PATH+'input.csv')

Synchronous models return a dictionary, while asynchronous models return an instance of the :py:class:`mlops_codex.base.MLOpsExecution`. This instance allows you to monitor the status and download the results, similar to how you would with training executions.

To use the models, you will need a `group token`, which is generated when creating the group (see :ref:`connecting_to_mlops:creating a group`). You can set this token by adding it to the `MLOPS_GROUP_TOKEN` environment variable, using the :py:meth:`mlops_codex.model.MLOpsModel.set_token` method, or passing it directly in each :py:meth:`mlops_codex.model.MLOpsModel.predict` call.

In many cases, you may need to use your model outside of a Python environment, often by sharing it through a REST API. To facilitate this, you can access the :py:attr:`mlops_codex.model.MLOpsModel.docs` attribute to share an OpenAPI Swagger page, or use the :py:meth:`mlops_codex.model.MLOpsModel.generate_predict_code` method to generate sample request code for your model.

Disabling your model
----------------

Disabling a model means you will no longer be able to perform certain operations on it. Before proceeding, please ensure you have confirmation from your team regarding the permission to perform this operation.
To disable a model, you can use the :py:meth:`mlops_codex.model.MLOpsModel.disable` method.

.. code-block:: python

    model.disable()


Deleting your model
----------------

Deleting a model will make it unavailable. Before proceeding, please confirm with your team that you have permission to carry out this operation.
To delete a model, you can use the :py:meth:`mlops_codex.model.MLOpsModel.delete` method.

.. code-block:: python

    model.delete()


Monitoring your model
---------------------

Model monitoring involves tracking the model's performance in production to enable updates if it begins to make inaccurate predictions.

Currently, MLOps employs indirect monitoring. This means it observes the model's input in production and verifies its similarity to the training data.
When setting up the monitoring system, it is essential to identify which training process produced the model and which features are pertinent for monitoring.

We provide metrics such as the "Population Stability Index" (PSI and PSI average) and "SHapley Additive exPlanations" (SHAP and SHAP average).

Additionally, it is crucial to understand how to manage the features and the model effectively.

Production data is stored in its raw form, while training data is not (see training guide: :ref:`training_guide:Running a training execution`). Therefore, it is important to know the steps for processing raw production data to derive model features similar to those saved during training: :ref:`monitoring_parameters:Monitoring configuration`.

The first method to invoke is :py:meth:`mlops_codex.pipeline.MLOpsPipeline.register_monitoring_config`, which registers the monitoring configuration in the database.

.. code-block:: python

    # We can also add a monitoring configuration for the model

    PATH = './samples/monitoring/'

    model.register_monitoring(
        preprocess_reference='parse', # name of the preprocess function
        shap_reference='get_shap', # name of the shap function
        configuration_file=PATH+'configuration.json', # Path of the configuration file, but it could be a dict
        preprocess_file=PATH+'preprocess_sync.py', # Path of the preprocess script
        requirements_file=PATH+'requirements.txt' # Path of the requirements file                        
    )

Next, you can manually run the monitoring process, calling the method :py:meth:`mlops_codex.pipeline.MLOpsPipeline.run_monitoring`.

.. code-block:: python

    pipeline = MLOpsPipeline.from_config_file('./samples/pipeline-just-model.yml')
    pipeline.register_monitoring_config(
        directory = "./samples/monitoring", preprocess = "preprocess_async.py", preprocess_function = "score", 
        shap_function = "score", config = "configuration.json", packages = "requirements.txt"
    )
    pipeline.start()

Using with preprocess script
----------------------------

Sometimes, you might want to run a preprocessing script to adjust the model input data before executing it. With MLOps, you can easily do this.

You must first instantiate the :py:class:`mlops_codex.model.MLOpsModelClient`:

.. code-block:: python

    from mlops_codex.model import MLOpsModelClient
    model_client = MLOpsModelClient()


And now you just need to run the model using the preprocess script (check :ref:`preprocessing:Preprocessing module`).

For the **sync model**:

.. code-block:: python

    sync_model = model_client.get_model(group='groupname', model_id='M7abe6af98484948ad63f3ad03f25b6496a93f06e23c4ffbaa43eba0f6a1bb91')

    sync_model.set_token('29d9d82e09bb4c11b9cd4ce4e36e6c58') # token example

    data = {
     "mean_radius": 17.99,
     "mean_texture": 10.38,
     "mean_perimeter": 122.8,
     "mean_area": 1001.0,
     "mean_smoothness": 0.1184,
     "mean_compactness": 0.2776,
     "mean_concavity": 0.3001,
     "mean_concave_points": 0.1471,
     "mean_symmetry": 0.2419,
     "mean_fractal_dimension": 0.07871,
     "radius_error": 1.095,
     "texture_error": 0.9053,
     "perimeter_error": 8.589,
     "area_error": 153.4,
     "smoothness_error": 0.006399,
     "compactness_error": 0.04904,
     "concavity_error": 0.05373,
     "concave_points_error": 0.01587,
     "symmetry_error": 0.03003,
     "fractal_dimension_error": 0.006193,
     "worst_radius": 25.38,
     "worst_texture": 17.33,
     "worst_perimeter": 184.6,
     "worst_area": 2019.0,
     "worst_smoothness": 0.1622,
     "worst_compactness": 0.6656,
     "worst_concavity": 0.7119,
     "worst_concave_points": 0.2654,
     "worst_symmetry": 0.4601,
     "worst_fractal_dimension": 0.1189
    }

    sync_model.predict(data=data, preprocessing=sync_preprocessing)

And for the **async model**:

.. code-block:: python

    async_model = model_client.get_model(group='datarisk', model_id='Maa3449c7f474567b6556614a12039d8bfdad0117fec47b2a4e03fcca90b7e7c')
    PATH = './samples/asyncModel/'
    execution = async_model.predict(data=PATH+'input.csv', preprocessing=async_preprocessing)
    execution.wait_ready()
    execution.download_result()
