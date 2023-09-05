Treinando seu modelo
===================

Treinar seu modelo no Neomaril ajuda por centralizar os registros num único lugar, enquanto economiza tempo ao executar os experimentos em paralelo.

Além disso, o módulo de treinamento é conectado aos módulos de implantação (deploy) e monitoramento, o que facilita os próximos passos.


Tipos de treinamento
---------------

Primeiro, precisamos criar um experimento de treino. Nós agregamos múltiplas execuções de treinamento num único experimento de treino. Cada execução de treino pode eventualmente se tornar um modelo implantado ou não.

**Custom:** É quando o usuário define tanto o código, quanto o ambiente de treinamento.

**AutoML:** É quando o treinamento é pre-definido usando o módulo de AutoML. Os arquivos necessários são os dados de treinamento e alguns parâmetros de configuração para o pacote.

Criando o experimento de treino
--------------------------------

Nós podemos criar o experimento usando o método :py:meth:`neomaril_codex.training.NeomarilTrainingClient.create_training_experiment`.

.. code:: python

    # Creating a new training experiment
    training = training_client.create_training_experiment('Teste notebook Training custom', # Experiment name, this is how you find your model in MLFLow
                                            'Classification', # Model type. Can be Classification, Regression or Unsupervised
                                            'Custom', # Training type. Can be Custom or AutoML
                                            group='datarisk' # This is the default group. Create a new one when using for a new project
                                            )


O retorno desse método é a classe :py:class:`neomaril_codex.training.NeomarilTrainingExperiment`, que nos possibilita criar múltiplas execuções que funcionam como versões do experimento principal.
Então você precisa enviar um dataset e o código de treinamento para essa execução (se for um experimento do tipo Custom) ou simplesmente a configuração (para um experimento do AutoML).

Executando um treinamento
----------------------------

Para um experimento Custom você precisa ter uma função de entrada como essa:

.. code:: python

    def train_model(base_path):
        df = pd.read_csv(base_path+"/dados.csv")
        X = df.drop(columns=['target'])
        y = df[["target"]]
        
        pipe = make_pipeline(SimpleImputer(), XGBClassifier())
        auc = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
        f_score = cross_val_score(pipe, X, y, cv=5, scoring="f1")
        pipe.fit(X, y)

        results = pd.DataFrame({"pred": pipe.predict(X), "proba": pipe.predict_proba(X)[:,1]})
        results.proba.hist().get_figure().savefig(base_path+'/probas.png', format='png')
        
        return {"X_train": X, "y_train": y, "model_output": results, "pipeline": pipe, 'extras': [base_path+'/probas.png'],
                "metrics": {"auc": auc.mean(), "f1_score": f_score.mean()}}


O único parâmetro na função é o caminho para o arquivo com os dados. Desta forma, nós podemos executar o treinamento quando os arquivos forem enviados para o Neomaril.
No experimento de treino Custom você tem todo o controle do processo, podendo testar vários algoritmos, otimizações de hiperparâmetros e validar em múltiplos segmentos dos dados.
O importante é respeitar o padrão de retorno da função, onde esperamos receber as informações sobre essa versão do modelo final para fins de registro. O retorno deve ser um dicionário com as seguintes chaves:

- `X_train`: O dataframe que será usado para ajustar o modelo.
- `y_train`: O dataframe/array/series alvo (target) que será usado para ajustar o modelo.
- `model_output`: Um dataframe/array/series com as saídas (outputs) do modelo. Por exemplo, os valores preditos/probabilidades, classes ou qualquer outra informação útil. Essa informação deve estar na saída do modelo implantado (deployed) futuramente para ser usado no monitoramento.
- `pipeline`: A instância final do modelo ajustado. Idealmente deve ser um `Scikit-Learn Pipeline Class <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_, mas outras classes de algoritmos que tem o método *get_params* implementado também funcionam. Esse modelo é salvo no arquivo `model.pkl` usando `cloudpickle <https://github.com/cloudpipe/cloudpickle> _` ou com o método `save_model`, se a classe do algoritmo tem essa função.
- `extra`: Uma lista opcional de nomes para arquivos extras que são gerados no treinamento. Podem ser plots, datasets de validação, etc. Eles devem ser salvos no mesmo caminho que é informado como parâmetro da função.
- `metrics`: Um dicionário com cada chave representando uma métrica. Você pode usar qualquer valor para as chaves e pode salvar quantos valores desejar, entretanto o valor deve ser numérico. Exemplo: `{"auc_train": 0.7, "auc_test": 0.65}`

Além disso, também precisamos das informações do ambiente (versão do Python e pacotes requeridos).

Então podemos chamar o método :py:meth:`neomaril_codex.training.NeomarilTrainingExperiment.run_training`.

.. code:: python

    # With the experiment class we can create multiple model runs
    PATH = './samples/train/'

    run1 = training.run_training('First test', # Run name
                                PATH+'dados.csv', # Path to the file with training data
                                source_file=PATH+'app.py', # Path of the source file
                                requirements_file=PATH+'requirements.txt', # Path of the requirements file, 
    #                           env=PATH+'.env'  #  File for env variables (this will be encrypted in the server)
    #                           extra_files=[PATH+'utils.py'], # List with extra files paths that should be uploaded along (they will be all in the same folder)
                                training_reference='train_model', # The name of the entrypoint function that is going to be called inside the source file 
                                python_version='3.9', # Can be 3.7 to 3.10
                                wait_complete=True
    )

Para o AutoML nós só precisamos dos dados e dos parâmetros de configuração. Você pode checar a :doc:`automl_parameters` para uma explicação mais detalhada.

.. code:: python

    # Creating a new training experiment
    training = training_client.create_training_experiment('Teste notebook Training AutoML', # Experiment name
                                                        'Classification', # Model type. Can be Classification, Regression or Unsupervised
                                                        'AutoML', # Training type. Can be Custom or AutoML
                                                        group='datarisk' # This is the default group. Create a new one when using for a new project
                                                        )

    PATH = './samples/autoML/'

    run2 = training.run_training('First test', # Run name
                                PATH+'dados.csv', # Path to the file with training data
                                conf_dict=PATH+'conf.json', # Path of the configuration file
                                wait_complete=True
    )



Verificando os resultados da execução
------------------------------

O retorno do método :py:meth:`neomaril_codex.training.NeomarilTrainingExperiment.run_training` é uma instância da classe :py:class:`neomaril_codex.training.NeomarilTrainingExecution`.
Com essa classe nós podemos seguir a execução assíncrona daquela versão do experimento e verificar suas informações.

.. code:: python

    run1.get_status()

    #>>> {'trainingExecutionId': '3', 'Status': 'Running', 'Message': None}

    run1.execution_data

    #>>> {'TrainingHash': 'T48c2371e453418f9859aba957de85cbcf84928d62a048b48f0259b49054a639',
    #     'ExperimentName': 'Teste notebook Training custom',
    #     'GroupName': 'datarisk',
    #     'ModelType': 'Classification',
    #     'TrainingType': 'Custom',
    #     'ExecutionId': 3,
    #     'ExecutionState': 'Running',
    #     'RunData': {},
    #     'RunAt': '2023-05-25T17:37:07.8850840Z',
    #     'Status': 'Requested'}


Nós podemos inclusive baixar os resultados (arquivo do modelo e outros salvos com a chave `extra`).

.. code:: python

    run1.download_result()
    
    #>>> 2023-05-26 10:02:13.441 | INFO     | neomaril_codex.base:download_result:376 - Output saved in ./output_2.zip

Se o modelo estiver bom o suficiente podemos começar o processo de implantação.
