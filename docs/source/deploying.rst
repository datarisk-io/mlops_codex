Implantação (deploying) em produção
=======================

Quando implantamos um modelo no Neomaril, nós criamos uma API para que você consiga conectar seu modelo a outros serviços. Você pode também usar o Neomaril Codex para executar o modelo remotamente dentro de uma aplicação Python.


Preparando para produção
------------------------

A primeira coisa que precisamos é do script de escoragem. Assim como no treinamento, nós precisamos de uma função de entrada (entrypoint). Os parâmetros e retornos dessa função vão depender da operação do modelo.


**Sync model:** Abreviação para modelo síncrono. Esse é o modelo que mais se aproxima da ideia de operação em "tempo real". Este modelo espera um JSON e retorna também um JSON após alguns segundos.
A função de entrada (entrypoint) deve ser parecida com:

.. code:: python

    def score(data, base_path):
        model = load(base_path+"/model.pkl")

        df = pd.DataFrame(data=json.loads(data), index=[0])
        
        return {"pred": int(model.predict(df)), "proba": float(model.predict_proba(df)[0,1])}

O primeiro parâmetro é a entrada do modelo no formato JSON. Essa entrada é recebida como uma string JSON, então você pode decodificar (parse) da maneira que quiser.
O outro parâmetro é um caminho (string). Assim como no treinamento, ele é usado para localizar tanto os arquivos do modelo, quanto outros arquivos que você pode subir (upload) para o Neomaril (veja a próxima seção).
O retorno dessa função deve ser um dicionário que pode ser convertido em JSON, ou então uma string JSON válida.

Tenha em mente que alguns tipos de dados (por exemplo valores numpy `int64` e `float64`) não podem ser convertidos em JSON normalmente, então seu código deve gerenciar esse passo antes de retornar a resposta para o Neomaril.

**Async model:** Abreviação para modelo assíncrono. Esse tipo de modelo deve ser usado para cenários de escoragem em batch, onde são enviados arquivos com vários registros de uma vez. Como esse processo geralmente leva mais tempo para finalizar dependendo do tamanho dos arquivos de entrada, ele é executado de maneira assíncrona.
A função de entrada (entrypoint) deve ser parecida com:

.. code:: python

    def score(data_path, model_path):
    
        model = load(model_path+"/model.pkl")

        X = pd.read_csv(data_path+'/input.csv')
        df = X.copy()

        df['proba'] = model.predict_proba(X)[:,1]
        df['pred'] = model.predict(X)

        output = data_path+'/output.csv'

        df.to_csv(output, index=False)

        return output

O primeiro parâmetro dessa função é uma string com o caminho para encontrar os dados de entrada. Note que temos parâmetros diferentes para os caminhos da função, pois cada execução do modelo assíncrono é salva num local distinto. Além disso os arquivos enviados para o servidor (uploaded) quando implantando (deploying) o modelo são constantes.
Se você deseja manter seu código mais dinâmico (e não forçar um padrão para o nome dos arquivos), você pode usar a variável de ambiente `inputFileName`, que terá o mesmo valor do nome do arquivo enviado (uploaded) para aquela execução.
Você deve salvar o resultado no mesmo caminho usado para os arquivos de entrada. E o retorno dessa função deve ser o caminho completo do resultado.


Implantando (deploying) seu modelo
--------------------

Com todos os arquivos prontos podemos fazer a implantação do modelo de duas formas.

- Usando o :py:meth:`neomaril_codex.training.NeomarilTrainingExecution.promote_model` para promover uma execução de treinamento bem-sucedida.

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



- Usando o :py:meth:`neomaril_codex.model.NeomarilModelClient.create_model` para implantar o modelo treinado fora do Neomaril.

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



Como você pode ver, implantar um modelo já treinado no Neomaril requer menos informações (os modelos vindo do AutoML requerem apenas 2 parâmetros).

Esses métodos retornam uma :py:class:`neomaril_codex.model.NeomarilModel`. Você pode usar o parâmetro *wait_for_ready* no método de implantação, ou chamar o método :py:meth:`neomaril_codex.model.NeomarilModel.wait_ready` para garantir que a instância :py:class:`neomaril_codex.model.NeomarilModel` está pronta para uso.
Nós vamos instalar as dependências do modelo (se você estiver promovendo um treinamento nós vamos usar as mesmas dependências usadas na execução do treinamento), e executar alguns testes. Para os modelos síncronos, é necessário um exemplo em JSON do esquema esperado para a API.

Se a implantação for bem-sucedida você pode já começar a usar seu modelo.

Usando seu modelo
---------------------

Nós podemos usar a mesma instância :py:class:`neomaril_codex.model.NeomarilModel` para chamar o modelo.

.. code:: python

    sync_model.predict({'key': 'value'})
    # >>> {'pred': 0, 'proba': 0.005841062869876623}
    
    execution = async_model.predict(PATH+'input.csv')
    # >>> 2023-05-26 12:04:14.714 | INFO     | neomaril_codex.model:predict:344 - Execution 5 started. Use the id to check its status.


Modelos síncronos retornar um dicionário e modelos assíncronos retorna uma :py:class:`neomaril_codex.base.NeomarilExecution` que pode ser usada para verificar o status e fazer o download do resultado, similar ao contexto da execução de treinamento.

Para usar os modelos você precisa de um `group token`, que é gerado no momento de criação do grupo (verifique :ref:`connecting_to_neomaril:creating a group`). Você pode adicionar esse token à variável de ambiente NEOMARIL_GROUP_TOKEN, usando o método :py:meth:`neomaril_codex.model.NeomarilModel.set_token`, ou então adicionar em cada chamada ao método :py:meth:`neomaril_codex.model.NeomarilModel.predict`.


A maior parte do tempo você precisará usar seu modelo fora do ambiente Python, compartilhando através da API REST.
Você pode chamar o atributo :py:attr:`neomaril_codex.model.NeomarilModel.docs` para compartilhar uma página no formato OpenAPI Swagger, ou então usar o método :py:meth:`neomaril_codex.model.NeomarilModel.generate_predict_code` para criar o código de exemplo de uma requisição para o modelo.


Monitorando seu modelo
---------------------

Monitorar o modelo significa entender como este está se comportanto em produção, de forma a entender se é o momento de atualiza-lo devido ao nível de predições erradas.

Atualmente, o Neomaril faz apenas o monitoramento indireto. Isso significa acompanhar a entrada do modelo em produção e verificar se está próxima dos dados apresentados no treinamento.
Então, quando configuramos o monitoramento nós precisamos saber quais dados de treinamento geraram o modelo, e quais features são relevantes para o processo de monitoramento.

Além disso, precisamos saber como lidar tanto com as features quanto com o modelo.

Os dados de produção são salvos em formato cru (raw), mas os dados de treinamento não (verifique :ref:`training_guide:Running a training execution`). Então, precisamos saber quais são os passos no processamento dos dados crus para obter as features do modelo, como foi feito durante o treinamento:

**TBD in the preprocess module.**

