Deploying to production
=======================

When deploying a model to Neomaril we create an API so you can connect your model with other services. You can also use Neomaril Codex to execute your model remotely inside a python application.


Preparing to production
------------------------

The first thing we need is the scoring script. Similar to the training we need a entrypoint function. The parameters and return of this function will depend on the model operation. 


**Sync model:** This is the "real time" model. The model will expect a JSON and return a JSON after a few seconds.
The entrypoint function should look like this:

.. code:: python

    # Function that describes the steps for Neomaril to execute the trained model.
    # The input parameters are:
    #   data:str
    #       The data that will be used by the model must arrive in string format
    #   base_path:str
    #       The path to the template file and other supplementary files
    def score(data:str, base_path:str): # It is the name of the function (score) that must be passed in the 'model_reference' field

        # Environment variables loaded from a user-supplied file in the 'env' field
        # my_var = os.getenv('MY_VAR')
        # if my_var is None:
        #   raise Exception("Could not find `env` variable")

        ## Neomaril environment variable with model file name
        # with open(base_path+os.getenv('modelFileName'), 'rb') as f:

        # Loads the already trained model to be run based on the model file passed as a parameter
        with open(base_path+"/model.pkl", 'rb') as f:
            model = load(f)

        # Build a DataFrame with the input data. The data arrives as a JSON string so we have to transform it into a dictionary
        df = pd.DataFrame(data=json.loads(data), index=[0])
        
        # Returns the results of running the model as a dictionary
        # It's important to note that in this case, as we're converting to JSON, we can't use numpy types, so we convert to pure int and float.
        return {"pred": int(model.predict(df)), "proba": float(model.predict_proba(df)[0,1])}

The first parameter is the JSON data that will be sent to the model (this comes as a JSON string, so you should parse it the way you want).
The other one is the path. Like the training you can use it to open the model files and other files you will upload (see next section).
The return of the function should be a dictionary that can be parsed to a JSON, or a already valid JSON string. 

Keep in mind that some data types (like numpy `int64` and `float64` values) cannot normally be parsed to JSON, so your code should handle that before returning the response to Neomaril. 

**Async model:** This is for batch scoring. We send files with usually a lot of records at once. Since this might take a while depeding on the file size, we run this asynchronously.
The entrypoint function should look like this:

.. code:: python

    # Function that describes the steps for Neomaril to run the trained model.
    # The input parameters are:
        # data_path : str
            # The path to the data that will be used by the model, which should arrive in string format
        # model_path : str
            # The path to the model file and other complementary files
    def score(data_path:str, model_path:str):## Environment variables loaded from a file supplied by the user in the 'env' field

        # my_var = os.getenv('MY_VAR')
        # if my_var is None:
        #    raise Exception("Could not find `env` variable")

        ## Environment variable loaded from Neomaril with model file name
        # with open(base_path+os.getenv('modelFileName'), 'rb') as f:

        # Loads the already trained model to be run based on the model file passed as a parameter
        with open(model_path+"/model.pkl", 'rb') as f:
            model = load(f)

        ## Environment variable loaded from Neomaril with database name
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

The first parameter is now also a path for the data. We have different path parameter because each async model execution is saved in a different place. And the files uploaded when deploying the model are kept the same every time.
If you want to keep your code more dynamic (and don't want to enforce a file name pattern) you can use the `inputFileName` env variable, that will be same as the filename uploaded for that execution.
You must save the result in the same path you got the input file. And the return of that function should be this full path.


Deploying your model
--------------------

With all files ready we can deploy the model in two ways.

- Using the :py:meth:`neomaril_codex.training.NeomarilTrainingExecution.promote_model` to promote a succeeded training execution.

.. code:: python

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



- Using the :py:meth:`neomaril_codex.model.NeomarilModelClient.create_model` to deploy a model trained outside Neomaril

.. code:: python
    
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
        python_version='3.9', # Can be 3.7 to 3.10
        operation="Sync", # Can be Sync or Async
        group='datarisk' # Model group (create one using the client)
    )


As you can see deploying a model already trained in Neomaril requires less information (the AutoML models require only 2 parameters).

Those methods return a :py:class:`neomaril_codex.model.NeomarilModel`. You can use the *wait_for_ready* parameter on the deployment method or call the :py:meth:`neomaril_codex.model.NeomarilModel.wait_ready` to make sure the :py:class:`neomaril_codex.model.NeomarilModel` instance is ready to use.
We will install the model depedencies (if you are promoting a training we will use the same as the training execution), and run some tests. For the sync models we require a sample JSON of the expected schema for the API.

If the deployment succeeds you can start using your model.

Using your model
---------------------

We can use the same :py:class:`neomaril_codex.model.NeomarilModel` instance to call the model.

.. code:: python

    sync_model.predict(data={'key': 'value'})
    # >>> {'pred': 0, 'proba': 0.005841062869876623}
    
    execution = async_model.predict(data=PATH+'input.csv')
    # >>> 2023-05-26 12:04:14.714 | INFO     | neomaril_codex.model:predict:344 - Execution 5 started. Use the id to check its status.


Sync models return a dictionary and async models return a :py:class:`neomaril_codex.base.NeomarilExecution` that you can use to check the status and download the result similiar to the training execution.

To use the models you need a `group token`, that is generated when creating the group (check :ref:`connecting_to_neomaril:creating a group`). You can add this token in the NEOMARIL_GROUP_TOKEN env variable, use the :py:meth:`neomaril_codex.model.NeomarilModel.set_token` method or add in each :py:meth:`neomaril_codex.model.NeomarilModel.predict` call.


Most of the time you might need to used your model outside a python environment, sharing it through a REST API.
You can call the :py:attr:`neomaril_codex.model.NeomarilModel.docs` attribute to share a OpenAPI Swagger page, or use the :py:meth:`neomaril_codex.model.NeomarilModel.generate_predict_code` method to create a sample request code to your model. 


Monitoring your model
---------------------

Model monitoring means keeping track of how the model is being used in production, so you can update it if it starts making bad predictions.

For now, Neomaril only does indirect monitoring. This means that Neomaril follows the input of the model in production and checks if it is close to the data presented to the model in training.
So, when configuring the monitoring system, we need to know which training generated that model and what features are relevant to monitoring the model.

We offer both "Population Stability Index" (PSI and PSI average) and "SHapley Additive exPlanations" (SHAP and SHAP average) metrics.

Besides that, we need to know how to correctly handle the features and the model. 

The production data is saved raw, and the training data is not (check :ref:`training_guide:Running a training execution`). So we need to know the steps in processing the raw production data to get the model features like the ones we saved during training: :ref:`monitoring_parameters:Monitoring configuration`

The first method you need to call is :py:meth:`neomaril_codex.pipeline.NeomarilPipeline.register_monitoring_config`, which is responsible for registering the monitoring configuration at the database.

.. code:: python
    # # We can also add a monitoring configuration for the model

    PATH = './samples/monitoring/'

    model.register_monitoring(
        preprocess_reference='parse', # name of the preprocess function
        shap_reference='get_shap', # name of the preprocess function
        configuration_file=PATH+'configuration.json', # Path of the configuration file, but it could be a dict
        preprocess_file=PATH+'preprocess_sync.py', # Path of the preprocess script
        requirements_file=PATH+'requirements.txt' # Path of the requirements file                        
    )
    # >>> 2023-10-26 09:18:46.940 | INFO     | neomaril_codex.model:register_monitoring:604 - Monitoring created - Hash: "M3aa182ff161478a97f4d3b2dc0e9b064d5a9e7330174daeb302e01586b9654c"

Next, you can manually run the monitoring process, calling the method :py:meth:`neomaril_codex.pipeline.NeomarilPipeline.run_monitoring`.

.. code:: python
    pipeline = NeomarilPipeline.from_config_file('./samples/pipeline-just-model.yml')
    pipeline.register_monitoring_config(directory = "./samples/monitoring", preprocess = "preprocess_async.py", preprocess_function = "score", 
                                        shap_function = "score", config = "configuration.json", packages = "requirements.txt")
    pipeline.start()

    # >>> 2023-10-25 16:13:01.706 | INFO     | neomaril_codex.pipeline:from_config_file:124 - Loading .env
    # >>> 2023-10-25 16:13:01.708 | INFO     | neomaril_codex.pipeline:__init__:43 - Loading .env
    # >>> 2023-10-25 16:13:01.709 | INFO     | neomaril_codex.pipeline:run_deploy:242 - Deploying scorer
    # >>> 2023-10-25 16:13:01.711 | INFO     | neomaril_codex.model:__init__:722 - Loading .env
    # >>> 2023-10-25 16:13:01.712 | INFO     | neomaril_codex.base:__init__:90 - Loading .env
    # >>> 2023-10-25 16:13:03.455 | INFO     | neomaril_codex.base:__init__:102 - Successfully connected to Neomaril
    # >>> 2023-10-25 16:13:04.849 | ERROR    | neomaril_codex.base:create_group:162 - {"Error":{"Type":"BadInput","Message":"Detail redacted as it may contain sensitive data. Specify \u0027Include Error Detail\u0027 in the connection string to include this information."}}
    # >>> 2023-10-25 16:13:04.850 | ERROR    | neomaril_codex.base:create_group:163 - Group already exist, nothing was changed.
    # >>> 2023-10-25 16:13:08.274 | INFO     | neomaril_codex.model:__upload_model:1015 - Model 'Teste' inserted - Hash: "Mc4f6403c5ab466f911c1cc6d2f22390fc5ab572337b42a7944fcc5d478849be"
    # >>> 2023-10-25 16:13:10.002 | INFO     | neomaril_codex.model:__host_model:1046 - Model host in process - Hash: Mc4f6403c5ab466f911c1cc6d2f22390fc5ab572337b42a7944fcc5d478849be
    # Wating for deploy to be ready.............
    # >>> 2023-10-25 16:15:28.933 | INFO     | neomaril_codex.model:get_model:820 - Model Mc4f6403c5ab466f911c1cc6d2f22390fc5ab572337b42a7944fcc5d478849be its deployed. Fetching model.
    # >>> 2023-10-25 16:15:28.936 | INFO     | neomaril_codex.model:__init__:69 - Loading .env
    # >>> 2023-10-25 16:15:33.139 | INFO     | neomaril_codex.pipeline:run_deploy:257 - Model deployement finished
    # >>> 2023-10-25 16:15:33.140 | INFO     | neomaril_codex.pipeline:run_monitoring:277 - Configuring monitoring
    # >>> 2023-10-25 16:15:33.142 | INFO     | neomaril_codex.model:__init__:69 - Loading .env
    # >>> 2023-10-25 16:15:37.849 | INFO     | neomaril_codex.model:register_monitoring:604 - Monitoring created - Hash: "Mc4f6403c5ab466f911c1cc6d2f22390fc5ab572337b42a7944fcc5d478849be"


Using with preprocess script
----------------------------

Sometimes you want to run a preprocess script to adjust the model input data before executing it. With Neomaril you can do it.

You must first instantiate the :py:class:`neomaril_codex.base.NeomarilExecution`:

.. code:: python

    model_client = NeomarilModelClient()
    # >>> 2023-10-26 10:26:42.351 | INFO     | neomaril_codex.model:__init__:722 - Loading .env
    # >>> 2023-10-26 10:26:42.352 | INFO     | neomaril_codex.base:__init__:90 - Loading .env
    # >>> 2023-10-26 10:26:43.716 | INFO     | neomaril_codex.base:__init__:102 - Successfully connected to Neomaril

And now you just need to run the model using the preprocess script (check :ref:`preprocessing:Preprocessing module`).
For the **sync model**:

.. code:: python

    sync_model = model_client.get_model(group='datarisk', model_id='M3aa182ff161478a97f4d3b2dc0e9b064d5a9e7330174daeb302e01586b9654c')

    sync_model.predict(data=sync_model.schema, preprocessing=sync_preprocessing)
    # >>> 2023-10-26 10:26:45.121 | INFO     | neomaril_codex.model:get_model:820 - Model M3aa182ff161478a97f4d3b2dc0e9b064d5a9e7330174daeb302e01586b9654c its deployed. Fetching model.
    # >>> 2023-10-26 10:26:45.123 | INFO     | neomaril_codex.model:__init__:69 - Loading .env
    # >>> {'pred': 0, 'proba': 0.005841062869876623}

And for the **async model**:

.. code:: python

    async_model = model_client.get_model(group='datarisk', model_id='Maa3449c7f474567b6556614a12039d8bfdad0117fec47b2a4e03fcca90b7e7c')

    PATH = './samples/asyncModel/'

    execution = async_model.predict(data=PATH+'input.csv', preprocessing=async_preprocessing)
    execution.wait_ready()
    # >>> 2023-10-26 10:26:51.460 | INFO     | neomaril_codex.model:get_model:820 - Model Maa3449c7f474567b6556614a12039d8bfdad0117fec47b2a4e03fcca90b7e7c its deployed. Fetching model.
    # >>> 2023-10-26 10:26:51.461 | INFO     | neomaril_codex.model:__init__:69 - Loading .env
    # >>> 2023-10-26 10:26:54.532 | INFO     | neomaril_codex.preprocessing:set_token:123 - Token for group datarisk added.
    # >>> 2023-10-26 10:26:55.955 | INFO     | neomaril_codex.preprocessing:run:177 - Execution '4' started to generate 'Db84e3baffc3457b9729f39f9f37aa1cd8aada89d3434ea0925e539cb23d7d65'. Use the id to check its status.
    # >>> 2023-10-26 10:26:55.956 | INFO     | neomaril_codex.base:__init__:279 - Loading .env
    # >>> 2023-10-26 10:30:12.982 | INFO     | neomaril_codex.base:download_result:413 - Output saved in ./result_preprocessing
    # >>> 2023-10-26 10:30:14.619 | INFO     | neomaril_codex.model:predict:365 - Execution '5' started. Use the id to check its status.
    # >>> 2023-10-26 10:30:14.620 | INFO     | neomaril_codex.base:__init__:279 - Loading .env

    execution.download_result()
    # >>> 2023-10-26 10:32:28.296 | INFO     | neomaril_codex.base:download_result:413 - Output saved in ./output.zip