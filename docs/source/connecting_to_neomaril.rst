Conectando ao Neomaril
======================

Para interagir com o Neomaril devemos acessar os clientes.

Atualmente temos duas opções :py:class:`neomaril_codex.training.NeomarilTrainingClient` e :py:class:`neomaril_codex.model.NeomarilModelClient`.

Vocês precisa da URL do servidor e um token válido. A melhor forma de configurar é usando um arquivo *.env* com essas variáveis de ambiente:

.. code::

    NEOMARIL_URL='https://neomaril.staging.datarisk.net'
    NEOMARIL_TOKEN='123'


Se você criar esse arquivo no mesmo diretório que está executando seu código, elas seram importadas automaticamente.

.. code:: python

    from neomaril_codex.training import NeomarilTrainingClient
    training_client = NeomarilTrainingClient()

    #>>> 2023-05-24 10:58:24.855 | INFO     | neomaril_codex.base:__init__:87 - Loading .env
    #>>> 2023-05-24 10:58:25.028 | INFO     | neomaril_codex.base:__init__:99 - Successfully connected to Neomaril




Criando um grupo
----------------

Grupos são usados para separar experimentos de treino e modelos que podem ter diferentes usuários finais.
Internamente, nós os usamos para organizar o sistema de arquivos e a rede numa configuração que nos possibilita criar processos isolados para cada grupo. Quando um grupo é criado, concomitantemente é gerado um token único, que deve ser usado para executar os modelos de modo seguro.

Todo recurso criado no Neomaril deve estar num grupo, então esse deve ser o primeiro passo na utilização da ferramenta.

Para criar um grupo você pode usar qualquer cliente, nós precisamos apenas do nome. Além disso também podemos adicionar uma descrição.

.. code:: python

   # Import the client
    from neomaril_codex.training import NeomarilTrainingClient

    training_client = NeomarilTrainingClient()

    model_client.create_group('nb_demo', # Group name
                    'Group for the demo' # A small description
                    )

    #>>> 2023-05-24 10:58:25.634 | INFO     | neomaril_codex.base:create_group:155 - Group 'nb_demo' inserted. Use the following token for scoring: 'f376c18092314246a432a2882c3cc8fd'. Carefully save it as we won't show it again.' 

    # We create a separate group token to be used in model predictions, so it can be shared with the clients
    # This token has a 1 year expiration date, to generate a new one use the refresh method

    model_client.refresh_group_token('nb_demo', # Group name
                                force=True # To force creating a new token even if the old is valid
                                )