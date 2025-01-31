Preprocessing data with MLOps Codex
=======================================

Now you can preprocess multiples data with MLOps Codex using the the new preprocessing.

With the :py:class:`mlops_codex.preprocessing.MLOpsPreprocessingClient` class give you the interaction between your code and our environment.

Hosting a preprocess script
---------------------------

Let's get started. First import and connect to MLOps:

.. code:: python

    from mlops_codex.preprocessing import MLOpsPreprocessingClient

    client = MLOpsPreprocessingClient()

Prepare your data. In this case, you must do the following:

.. code:: python
    # In case you just have one database:
    data = (<dataset_name>, <path_to_database>)

    # If you want to send multiple files:
    data = [
        (<dataset_name_1>, <path_to_database_1>),
        (<dataset_name_2>, <path_to_database_2>),
        (<dataset_name_3>, <path_to_database_3>),
        # ....
    ]

Note the data structure must be as above

Next, send the data to our server, by using the `.create()` method. Check the parameters doc.

To do it, you must do as following:

.. code:: python
    preprocess = client.create(
        name="test_preprocessing",
        group="groupname",
        schema_files_path=schemas,
        script_path=PATH+'app.py',
        entrypoint_function_name="build_df",
        python_version='3.9',
        requirements_path=PATH+'requirements.txt',
        extra_files=extra_files,
        wait_read=True
    )

The `wait_read` and `host` are optional parameters. They control the initial state of the preprocessing object.


Creating an execution
---------------------

To create a new execution, do as the following code:

.. code:: python
    run = preprocess.run(
        input_files=inputs,
        wait_read=True
    )


Download the result of your preprocessing
-----------------------------------------

The result of a preprocessing execution is a `preprocessed_data.parquet`. To download the result, check the example below:

.. code:: python

    run.download()
