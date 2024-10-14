Training your model
===================

Training your model in Neomaril helps by logging everything in a model registry, while also saving time by running experiments in parallel.

Also, the training module is connected to the deploying and monitoring module, so using the training in Neomaril saves time in the next steps.


Training types
---------------

First we need to create a training experiment. We aggregate multiple train runs into one training experiment. Each train run can eventually became a deployed model or not.

**Custom:** Is when the user will define the whole training code and define the training environment.

**AutoML:** Is when the training is pre defined using the AutoML module. The required files are the training data and some configuration parameters for the package.

**External:** Is when you perform the model training on your machine or any other place, external to Neomaril. You need to upload it, if you want to monitor your model in our environment.

Creating the training experiment
--------------------------------

We can create the experiment using the :py:meth:`neomaril_codex.training.NeomarilTrainingClient.create_training_experiment` method.

.. code:: python

    # Start the client
    training_client =  NeomarilTrainingClient()

    # Creating a new training experiment
    training = training_client.create_training_experiment(
        experiment_name='Teste notebook', # Experiment name, this is how you find your model in MLFLow
        model_type='Classification', # Model type. Can be Classification, Regression or Unsupervised
        group='datarisk' # This is the default group. Create a new one when using for a new project
    )


The return of the method is a :py:class:`neomaril_codex.training.NeomarilTrainingExperiment`, where you can create multiple executions that works like versions of the main experiment.
Then you need to actually upload a dataset and a training code for your execution (if is a Custom experiment) or the configuration (for the AutoML experiment).

Running a training execution
----------------------------

For the custom experiment you need a entrypoint function like this:

.. code:: python
    import pandas as pd
    from lightgbm import LGBMClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score
    import os
    
    # Function that provides the steps to train the model based on a given dataset.
    # Its name (train_model) must be passed in the 'training_reference' field
    # Its parameter (base_path) must contain the folder path for the files that will be used.
    # An example for its value is: "/path/to/treino/customizado/experimento1"
    def train_model(base_path): 

        # Environment variables loaded from a user-supplied file in the 'env' field
        # my_var = os.getenv('MY_VAR')
        # if my_var is None:
        #   raise Exception("Could not find `env` variable")

        ## Environment variable loaded from Neomaril with database name (used alternatively to line 61)
        # df = pd.read_csv(base_path+'/'+os.getenv('inputFileName'))
        df = pd.read_csv(base_path+"/data.csv")    # Load the database (data.csv) which must have the same name
                                                    # file sent to Neomaril in the 'train_data' field
        
        # The data sent must be the complete data for training and validation (excluding the validation sample),
        # it is at the user's discretion how to treat the data here
        X = df.drop(columns=['target'])             # Separates the database from the column with the targets
        y = df[["target"]]                          # Saves the target in a dataframe
        
        pipe = make_pipeline(SimpleImputer(), LGBMClassifier())     # Define the steps to train the model
        
        # In this example we used cross-validation, but this is at the user's discretion
        auc = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")  # Validation of results using the 'auc' metric
        f_score = cross_val_score(pipe, X, y, cv=5, scoring="f1")   # Validation of results using the 'f1' metric
        pipe.fit(X, y)  # Train the model

        # Build the DataFrame with the results
        results = pd.DataFrame({"pred": pipe.predict(X), "proba": pipe.predict_proba(X)[:,1]})  
        
        # Returns the training results according to the parameters expected by Neomaril
        return {"X_train": X, "y_train": y, "model_output": results, "pipeline": pipe, 
                "metrics": {"auc": auc.mean(), "f1_score": f_score.mean()}}


The only parameter on the function is the path for the data file. This way we can execute it when the files are uploaded to Neomaril.
In the custom training experiment you can do whatever you want, test multiple algorithms, optimize hyperparameters, validate on multiple segments of the data.
The important thing is the return of the function, where we get information about the final model of this version so we can log it. The return must be a dictionary with the following keys:

- `X_train`: The dataframe that will be used to fit the model.
- `y_train`: The target dataframe/array/series that will be used to fit the model.
- `model_output`: A dataframe/array/series with outputs of the model. This can be the predicted values/probabilities, classes or any other useful information. This information needs to be in the output of the future deployed model to be used in the monitoring
- `pipeline`: The final fitted model instance. Ideally it should be a `Scikit-Learn Pipeline Class <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_, but any other algorithm class that has the *get_params* method implemented works. This will be saved as `model.pkl` with `cloudpickle <https://github.com/cloudpipe/cloudpickle> _` or with the `save_model` method if the algorithm class has that.
- `extra`: A optional list of filenames for extra files that are generated in the training. This can be plots, validation datasets, etc. They need to be saved in the same path that is provided as the function parameter.
- `metrics`: A dictionary with each key as a metric. You can use any name for the metric key and save as many as you want, but the value must be numeric. Eg: `{"auc_train": 0.7, "auc_test": 0;65}`

Besides that we also need the information for the environment (python version and package requirements). 

Then we can call the :py:meth:`neomaril_codex.training.NeomarilTrainingExperiment.run_training` method.

.. code:: python

    # With the experiment class we can create multiple model runs
    PATH = './samples/train/'

    run1 = training.run_training(
        run_name='First test', # Run name
        train_data=PATH+'dados.csv', # Path to the file with training data
        training_type='Custom', # Training type. Can be External, Custom or AutoML
        source_file=PATH+'app.py', # Path of the source file
        requirements_file=PATH+'requirements.txt', # Path of the requirements file, 
        #env=PATH+'.env'  #  File for env variables (this will be encrypted in the server)
        #extra_files=[PATH+'utils.py'], # List with extra files paths that should be uploaded along (they will be all in the same folder)
        training_reference='train_model', # The name of the entrypoint function that is going to be called inside the source file 
        python_version='3.9', # Can be 3.8 to 3.10
        wait_complete=True
    )

For the AutoML we just need the data and the configuration parameters. You can check the :doc:`automl_parameters` for more details. 

.. code:: python

    PATH = './samples/autoML/'

    run2 = training.run_training(
        run_name='First test', # Run name
        training_type='Custom', # Training type. Can be External, Custom or AutoML
        train_data=PATH+'dados.csv', # Path to the file with training data
        conf_dict=PATH+'conf.json', # Path of the configuration file
        wait_complete=True
    )

See the example below, using a python script to perform and save an External training:

.. code:: python
    from neomaril_codex.training import NeomarilTrainingClient
    import pandas as pd
    from lightgbm import LGBMClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score
    import matplotlib.pyplot as plt

    # Start the model client
    client = NeomarilTrainingClient()

    # Create an experiment
    training = client.create_training_experiment('Teste', 'Classification', group='datarisk')

    # Your variables
    base_path = './samples/train/'
    df = pd.read_csv(base_path+"/dados.csv")
    X = df.drop(columns=['target'])
    y = df[["target"]]


    plt.scatter(df["mean_radius"], df["mean_texture"])

    # Graph Title
    plt.title("Relação entre mean_radius e mean_texture")

    # Config axis
    plt.xlabel("mean_radius")
    plt.ylabel("mean_texture")

    fig = plt.gcf()

    # Plot
    plt.show()

    # Build a pipeline
    pipe = make_pipeline(SimpleImputer(), LGBMClassifier(force_col_wise=True))

    # log the model and save the metrics and model output
    with training.log_train('Teste 1', X, y) as logger:
        pipe.fit(X, y)
        logger.save_model(pipe)

        logger.add_extra('./extra.txt')

        logger.save_and_add_plot(fig, 'graphic1')
        model_output = pd.DataFrame({"pred": pipe.predict(X), "proba": pipe.predict_proba(X)[:,1]})
        logger.save_model_output(model_output)

        auc = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
        f_score = cross_val_score(pipe, X, y, cv=5, scoring="f1")
        logger.save_metric(name='auc', value=auc.mean())
        logger.save_metric(name='f1_score', value=f_score.mean())

        logger.set_python_version('3.10')


Checking the execution results
------------------------------

The return of the :py:meth:`neomaril_codex.training.NeomarilTrainingExperiment.run_training` is a :py:class:`neomaril_codex.training.NeomarilTrainingExecution` instance
With this class we can follow the asynchronous execution of that experiment version and check information on it. 

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


We can also download the results (model file and files saved in the `extra` key)

.. code:: python

    run1.download_result()
    
    #>>> 2023-05-26 10:02:13.441 | INFO     | neomaril_codex.base:download_result:376 - Output saved in ./output_2.zip

If the model is good enough we can start the deploying process.
