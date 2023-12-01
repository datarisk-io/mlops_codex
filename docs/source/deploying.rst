Deploying to production
=======================

When deploying a model to Neomaril we create an API so you can connect your model with other services. You can also use Neomaril Codex to execute your model remotely inside a python application.


Preparing to production
------------------------

The first thing we need is the scoring script. Similar to the training we need a entrypoint function. The parameters and return of this function will depend on the model operation. 


**Sync model:** This is the "real time" model. The model will expect a JSON and return a JSON after a few seconds.
The entrypoint function should look like this:

.. code:: python

    # Função que descreve os passos para o Neomaril executar o modelo treinado.
    # Os parâmetros de entrada são:
    #   data : str
    #       Os dados que serão usados pelo modelo que deverão chegar no formato string
    #   base_path : str
    #       O caminho para o arquivo do modelo e outros arquivos complementares
    def score(data:str, base_path:str): # O nome da função (score) é que deve ser passado no campo 'model_reference'

        ## Variáveis de ambiente carregadas de um arquivo fornecido pelo usuário no campo 'env'
        # my_var = os.getenv('MY_VAR')
        # if my_var is None:
        #    raise Exception("Could not find `env` variable")

        ## Variável de ambiente do Neomaril com nome do arquivo do modelo
        # with open(base_path+os.getenv('modelFileName'), 'rb') as f:

        # Carrega o modelo já treinado para ser executado com base no arquivo de modelo passado como parâmetro
        with open(base_path+"/model.pkl", 'rb') as f:
            model = load(f)

        # Constrói um DataFrame com os dados de entrada. Os dados chegam como um JSON string então temos que tranforma-los em um dicionário
        df = pd.DataFrame(data=json.loads(data), index=[0])
        
        # Retorna os resultados da execução do modelo como um dicionário
        # Importante que nesse caso como vamos converter para um JSON não podemos usar os tipos do numpy, então convertemos para int e float puros.
        return {"pred": int(model.predict(df)), "proba": float(model.predict_proba(df)[0,1])}

The first parameter is the JSON data that will be sent to the model (this comes as a JSON string, so you should parse it the way you want).
The other one is the path. Like the training you can use it to open the model files and other files you will upload (see next section).
The return of the function should be a dictionary that can be parsed to a JSON, or a already valid JSON string. 

Keep in mind that some data types (like numpy `int64` and `float64` values) cannot normally be parsed to JSON, so your code should handle that before returning the response to Neomaril. 

**Async model:** This is for batch scoring. We send files with usually a lot o records at once. Since this might take a while depeding on the file size, we run this asynchronously.
The entrypoint function should look like this:

.. code:: python

    # Função que descreve os passos para o Neomaril executar o modelo treinado.
    # Os parâmetros de entrada são:
    #   data_path : str
    #       O caminho para os dados que serão usados pelo modelo que deverão chegar no formato string
    #   model_path : str
    #       O caminho para o arquivo do modelo e outros arquivos complementares
    def score(data_path:str, model_path:str):## Variáveis de ambiente carregadas de um arquivo fornecido pelo usuário no campo 'env'

        # my_var = os.getenv('MY_VAR')
        # if my_var is None:
        #    raise Exception("Could not find `env` variable")

        ## Variável de ambiente carregada do Neomaril com nome do arquivo do modelo
        # with open(base_path+os.getenv('modelFileName'), 'rb') as f:

        # Carrega o modelo já treinado para ser executado com base no arquivo de modelo passado como parâmetro
        with open(model_path+"/model.pkl", 'rb') as f:
            model = load(f)

        ## Variável de ambiente carregada do Neomaril com nome da base de dados
        # X = pd.read_csv(data_path+'/'+os.getenv('inputFileName'))

        # Carrega os dados da base de entrada do arquivo para um DataFrame
        X = pd.read_csv(data_path+"/dados.csv")

        df = X.copy()   # Cria uma cópia do DataFrame com os dados de entrada

        df['proba'] = model.predict_proba(X)[:,1]   # Calcula a predição de cada entrada da tabela de dados
        df['pred'] = model.predict(X)               # Calcula a probabilidade de cada entrada da tabela de dados

        # Cria o caminho com o nome do arquivo de output, nesse caso 'output.csv'. Importante que esse arquivo deve ser salvo no mesmo caminho que os dados que foram enviados.
        output = data_path+'/output.csv'

        # Transforma o DataFrame, com a predição e a probabilidade, para csv colocando no arquivo do caminho do output
        df.to_csv(output, index=False)

        # Retorna o caminho com o arquivo com os resultados da execução do modelo
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
    model = custom_run.promote_model('Teste notebook promoted custom', # model_name
                                    'score', # name of the scoring function
                                    PATH+'app.py', # Path of the source file
                                    schema=PATH+'schema.json', # Path of the schema file, but it could be a dict (only required for Sync models)
        #                           env=PATH+'.env'  #  File for env variables (this will be encrypted in the server)
        #                           extra_files=[PATH+'utils.py'], # List with extra files paths that should be uploaded along (they will be all in the same folder)
                                    operation="Sync" # Can be Sync or Async
    )

    # Promoting an AutoML training execution
    model = automl_run.promote_model('Teste notebook promoted autoML', # model_name
                                     operation="Async" # Can be Sync or Async
    )



- Using the :py:meth:`neomaril_codex.model.NeomarilModelClient.create_model` to deploy a model trained outside Neomaril

.. code:: python
    
    # Deploying a new model
    model = client.create_model('Teste notebook Sync', # model_name
                                'score', # name of the scoring function
                                PATH+'app.py', # Path of the source file
                                PATH+'model.pkl', # Path of the model pkl file, 
                                PATH+'requirements.txt', # Path of the requirements file, 
                                schema=PATH+'schema.json', # Path of the schema file, but it could be a dict (only required for Sync models)
    #                           env=PATH+'.env'  #  File for env variables (this will be encrypted in the server)
    #                           extra_files=[PATH+'utils.py'], # List with extra files paths that should be uploaded along (they will be all in the same folder)
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

    sync_model.predict({'key': 'value'})
    # >>> {'pred': 0, 'proba': 0.005841062869876623}
    
    execution = async_model.predict(PATH+'input.csv')
    # >>> 2023-05-26 12:04:14.714 | INFO     | neomaril_codex.model:predict:344 - Execution 5 started. Use the id to check its status.


Sync models return a dictionary and async models return a :py:class:`neomaril_codex.base.NeomarilExecution` that you can use to check the status and download the result similiar to the training execution.

To use the models you need a `group token`, that is generated when creating the group (check :ref:`connecting_to_neomaril:creating a group`). You can add this token in the NEOMARIL_GROUP_TOKEN env variable, use the :py:meth:`neomaril_codex.model.NeomarilModel.set_token` method or add in each :py:meth:`neomaril_codex.model.NeomarilModel.predict` call.


Most of the time you might need to used your model outside a python environment, sharing it through a REST API.
You can call the :py:attr:`neomaril_codex.model.NeomarilModel.docs` attribute to share a OpenAPI Swagger page, or use the :py:meth:`neomaril_codex.model.NeomarilModel.generate_predict_code` method to create a sample request code to your model. 


Monitoring your model
---------------------

Model monitoring means keeping track with how the model is being used in production so we can update the model if it start making bad predictions.

For now Neomaril only does indirect monitoring, that means following the input of the model in production and checking if is close to the data presented to the model in training.
So when configure the monitoring we need to know which training generated that model and what features are relevant to monitoring the model.

Besides we need to know how to handle the features and the model. 

The production data is saved raw, and the training data is not (check :ref:`training_guide:Running a training execution`). So we need to know the steps in processing the raw data to get the model features like the ones we saved during training:

**TBD in the preprocess module.**

