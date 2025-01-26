Preprocessing database with MLOps Codex
=======================================

Now you can preprocess multiples database with MLOps Codex using the the new

With the :py:class:`mlops_codex.preprocessing.MLOpsPreprocessingAsyncV2Client` class give you the interaction between your code and our environment.

With the :py:class:`mlops_codex.preprocessing.MLOpsPreprocessingAsyncV2` class you have access to different preprocessing attributes and how to manipulate them


Hosting a preprocess script
---------------------------

Let's get started. First import and connect to MLOps:

.. code:: python

    from mlops_codex.preprocessing import MLOpsPreprocessingAsyncV2Client

    client = MLOpsPreprocessingAsyncV2Client()

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
        wait_read=True
    )

The `wait_read` and `host` are optional parameters. They control the initial state of the preprocessing object.

This method returns a :py:class:`mlops_codex.preprocessing.MLOpsPreprocessingAsyncV2`.

Compatibility between the classes
---------------------------------
Both the :py:class:`mlops_codex.preprocessing.MLOpsPreprocessingAsyncV2` and :py:class:`mlops_codex.preprocessing.MLOpsPreprocessingAsyncV2Client` share the same interface.

Because that, most of the methods you perform via the client, you can perform using the preprocessing class.

The advantage? You don't need to keep your mind the preprocessing hash during the time you're working in your project.

However, the instructions will be lost after you restart your kernel! That's why we still recommend you to keep tracking of the preprocessing hashes and ids.

Creating an execution
---------------------

To create a new execution, do as the following code:

.. code:: python
    preprocess.run(
        input_data=inputs,
        wait_read=True
    )

Note: the inputs must have the same name!

Also, notice you can run using the client:

.. code:: python
    client.run(
        preprocessing_script_hash=preprocessing_hash
        input_data=inputs,
    )

In this case, you run, but you can't wait it for ready, so you must check the status:

.. code:: python
    client.execution_status(
        preprocessing_script_hash=preprocessing_script_hash,
        execution_id=execution_id
    )

Download the result of your preprocessing
-----------------------------------------

The result of a preprocessing execution is a `preprocessed_data.parquet`. To download the result, check the example below:

.. code:: python

    preprocess.download(execution_id=execution_id)

Again, you can use the client interface:

.. code:: python
    client.execution_status(
        download=preprocessing_script_hash,
        execution_id=execution_id
    )

Both methods contains a `path` parameter. If you change it, the local where it will be saved. Default is the parent directory.

Preprocessing MLOps datasets
----------------------------

The Dataset MLOps codex give you an interface to host and run preprocessing.

Given a dataset, you can perform the following code:

.. code:: python
    preproc = dataset.host_preprocessing(
        name="preprocessing_from_dataset",
        group="groupname",
        script_path=PATH+'app.py',
        entrypoint_function_name="build_df",
        python_version='3.9',
        requirements_path=PATH+'requirements.txt',
    )

Note, you don't have flow control in this case. That's why, the preprocessing script execution will be hosted and you'll wait until it is Succeeded or Failed.

Because it returns a :py:class:`mlops_codex.preprocessing.MLOpsPreprocessingAsyncV2`, you do the others operations.

To run a preprocessing script execution, you can do as following:

.. code:: python
    dataset.run_preprocess(
        preprocessing_script_hash=preprocessing_script_hash,
        execution_id=execution_id
    )

Note, if you want to perform it, you may have the preprocessing script execution and the execution id. Without that, it won't be possible to run the script.
