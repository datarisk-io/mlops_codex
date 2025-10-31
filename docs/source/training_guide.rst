Training your model
===================

Training your model in MLOps provides several benefits, including logging all activities in a model registry and improving efficiency by enabling parallel experiment runs.

Additionally, the training module is integrated with the deployment and monitoring modules, allowing you to seamlessly transition from training to deployment and monitoring, saving valuable time in the overall process.

First we need to create a training experiment. We aggregate multiple train runs into one training experiment. Each train run can eventually became a deployed model or not.

A training experiment requires the following key components:

- **Experiment Name**: A unique identifier for your training experiment.
- **Training Type**: The method used to train the model.
- **Model Type**: The type of model being trained.


Training types
---------------

The available options for training types are:

- **Custom:** Is when the user will define the whole training code and define the training environment.

- **AutoML:** Is when the training is pre defined using the AutoML module. The required files are the training data and some configuration parameters for the package.

- **External:** Is when you perform the model training on your machine or any other place, external to MLOps. You need to upload it, if you want to monitor your model in our environment.


Model Type
----------

The  parameter specifies the type of model you are training.

The available options for model types are:

- **Classification:** Used for training models that classify data into predefined categories. It is typically used in tasks such as image classification, text categorization, or binary classification.

- **Regression:** Used for training models that predict continuous numeric values. This model type is suitable for tasks like predicting prices, sales, or any task where the output is a continuous variable.

- **Unsupervised:** Used for training models that discover patterns in data without labeled outcomes. Common use cases include clustering, anomaly detection, and dimensionality reduction.


Creating the training experiment
--------------------------------

To create a new experiment, you must access the interface provided as shown in the example below.

.. code-block:: python

    from mlops_codex.administrator import DatariskMLops

    client = DatariskMLops(
        email="<email>",
        password="<password>",
        tenant="<tenant>"
    )

    experiment = client.train.setup_project_experiment(
        experiment_name='experimentName',
        model_type='Classification',
        group='<group>',
    )

The method returns an instance of the class :py:class:`mlops_codex.train.models.MLOpsExperiment`, which allows you to create multiple executions that function as versions of the main experiment.


Train Models
--------------

The datarisk-mlops-codex provides custom classes for each training type. Each class has its own necessary and optional files, allowing for better user control.


Running a custom training execution
~~~~~~~~~~~~~~~

For the custom experiment you need a entrypoint function like this:

.. code-block:: python

    import pandas as pd
    from lightgbm import LGBMClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score
    import os

    def train_model(df: pd.DataFrame, base_path: str): # The function name (train_model) should be passed in the 'Method to be called' field
        """
        Function used to train the model based on a provided dataset.
        This function should structure the steps that MLOps will have to execute in order to return the
        set of information resulting from the model training.

        In the function, the user can use environment variables loaded from a .env file,
        as shown in the code on lines 59-62.
        If the user does not want to keep the dataset name fixed, MLOps loads the name of the file
        in the environment variable (example on lines 64-65):
        inputFileName : str
            This variable contains the name of the dataset file that was uploaded.

        Parameters
        ---------
        df: pd.Dataframe
            The pandas dataframe that will be manipulated.
            This value is mandatory.

        base_path : str
            The folder path for the files that will be used.
            The user can use a default value for local tests, but in MLOps, the remote file path will be used.
            For example: "/path/to/custom_training/experiment1"

        Returns
        -------
        dict:
            A dictionary containing the following keys:
            X_train: DataFrame
                The data that will be used to train the model.
            y_train: DataFrame
                The dataframe/array/series of targets that will be used to train the model.
            model_output: DataFrame
                The dataframe/array/series with the results from the trained model.
                This could be predicted values/probabilities, classes, or any other useful information.
                This information must be in the model output to be used in future monitoring.
            pipeline: Pipeline
                The instance of the final trained model.
                Ideally, it should be a Scikit-Learn Pipeline class, but any algorithm class that
                implements the get_params method will work.
                This will be saved as model.pkl with cloudpickle <https://github.com/cloudpipe/cloudpickle> or using the
                save_model method if the algorithm class supports it.
            metrics: dict
                A dictionary with each key being a metric name.
                The user can use any name for the metric key and save as many as desired,
                but the value should be numeric.
                For example: {"auc_train": 0.7, "auc_test": 0.65}
            extra: string list
                An optional list of filenames for additional files generated during training.
                These could be graphs, validation sets, etc. They need to be saved in the same path (base_path)
                provided as a parameter to the function.
        """

        ## Environment variables loaded from a file provided by the user in the field
        ## 'File with environment variables' in step 3 (Optional additional files)
        # my_var = os.getenv('MY_VAR')
        # if my_var is None:
        #    raise Exception("Could not find `env` variable")

        X = df.drop(columns=['target'])             # Separates the data from the column with the targets
        y = df[["target"]]                          # Saves the targets in a DataFrame

        pipe = make_pipeline(SimpleImputer(), LGBMClassifier())     # Defines the steps to train the model

        # In this example, we use cross-validation, but this is at the user's discretion
        auc = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")  # Validation of results using the 'auc' metric
        f_score = cross_val_score(pipe, X, y, cv=5, scoring="f1")   # Validation of results using the 'f1' metric
        pipe.fit(X, y)  # Train the model

        # Build the DataFrame with the results
        results = pd.DataFrame({"pred": pipe.predict(X), "proba": pipe.predict_proba(X)[:,1]})

        # Returns the training results according to the parameters expected by MLOps
        return {"X_train": X, "y_train": y, "model_output": results, "pipeline": pipe,
                "metrics": {"auc": auc.mean(), "f1_score": f_score.mean()}}

The function accepts two parameters: the dataframe containing the data and the path to the data file. This allows the function to be executed when the files are uploaded to MLOps.
In the custom training experiment, you have full flexibility to experiment with different algorithms, optimize hyperparameters, and validate the model on multiple segments of the data.
The critical part is the function's return value, which provides information about the final model for logging and monitoring purposes. The return must be a dictionary containing the following keys:

- **X_train**: The dataframe that will be used to fit the model.
- **y_train**: The target dataframe/array/series that will be used to fit the model.
- **model_output**: A dataframe/array/series with the outputs of the trained model, such as predicted values, probabilities, or classes. This information must be included in the output of the deployed model to facilitate monitoring.
- **pipeline**: The final, fitted model instance. Ideally, this should be a `Scikit-Learn Pipeline Class <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_, but any other algorithm class that implements the get_params method will also work. This will be saved as model.pkl using `cloudpickle <https://github.com/cloudpipe/cloudpickle>`_ or the save_model method, if the algorithm class provides that.
- **extra (optional)**: A list of filenames for any additional files generated during training, such as plots, validation datasets, or logs. These files should be saved in the same path provided as the function parameter.
- **metrics**: A dictionary with metric names as keys and numeric values as the metrics (e.g., {"auc_train": 0.7, "auc_test": 0.65}). You can include as many metrics as needed, but each value must be a number.

Additionally, we also need environment information, such as the Python version and package requirements, to ensure compatibility and reproducibility of the training process.

This structure ensures that all relevant details are logged and available for monitoring in MLOps, making the model's lifecycle transparent and manageable.

.. code-block:: python

    # With the experiment class we can create multiple model runs
    train_type = CustomTrain(
        training_reference='train_model',
        run_name='more test',
        python_version='3.10',
        input_data=PATH + 'dados.csv',
        source=PATH + 'app.py',
        requirements=PATH + 'requirements.txt',
    )

    exe = client.train.run(
        experiment=experiment,
        train_type=train_type,
        wait_ready=True
    )

    print(exe.status)

Running an AutoML training execution
----------------------------

For the AutoML we just need the data and the configuration parameters. You can check the :doc:`automl_parameters` for more details. 

.. code-block:: python

    train_type = AutoMLTrain(
        run_name='autoMLTest',
        input_data=PATH + 'dados.csv',
        configuration=PATH + 'conf.json',
    )

    exe = client.train.run(
        experiment=experiment,
        train_type=train_type,
        wait_ready=True
    )

    print(exe.status)

See an example of the a configuration file:

.. code-block:: json

    {
        "train_data": {
            "file_type": "csv",
            "sep": ",",
            "file_name": "dados.csv"
        },
        "model_flow": "classification",
        "target": "target",
        "iterations": 1,
        "metric": "ks",
        "split_type": "random",
        "stages": {
            "models": ["logreg"]
            }
    }


Saving a local training to MLOps
----------------------------

See the example below, using a python script to perform and save an External training:

.. code-block:: python

    train_type = External(
        run_name='externalTest'
        python_version='3.10'
        features=PATH + 'features.parquet'
        target=PATH + 'target.parquet'
        output=PATH + 'output.parquet'
        metrics=PATH + 'metrics.json'
        model=PATH + 'model.pkl'
        requirements=PATH + 'requirements.txt'
        parameters=PATH + 'params.json'
    )

    exe = client.train.run(
        experiment=experiment,
        train_type=train_type,
        wait_ready=True
    )

    print(exe.status)


Downloading result
------------------------------

You can download the result of a training execution

.. code-block:: python
    exe.download()
