Connecting to a Data Source
===========

It is possible to connect a cloud provider to MLOps. The cloud provider is a source of data, where you can store and get the data that is saved data to create/perform models.

Currently, MLOps supports the following providers:

* Google GCP
* AWS S3
* Azure Blob Storage

Register a data source
----------------------

To register your data source to MLOps, you must have the provider credentials access json, data source name and group name:

.. code-block:: python

    client = MLOpsDataSourceClient()
    client.register_datasource(
        datasource_name='MyDataSourceName',
        provider='GCP',
        cloud_credentials='./gcp_credentials.json',
        group='my_group'
    )


If you already have a registered data source and want to get that, you can do as following:

.. code-block:: python

    datasource = client.get_datasource(datasource_name='testeDataSource', provider='GCP', group='datarisk')


Once you're connected to MLOps and registered your data source, you can list the available data sources:

.. code-block:: python

    client.list_datasource(provider='GCP', group='datarisk')


Importing a data set 
--------------------

Now, you already have access to a data source, it allows you to import a data set to your data source. It is mandatory that you register a datasource so that you can import your dataset into it.

You can import a data set via url:

.. code-block:: python

    dataset_uri = 'https://storage.cloud.google.com/projeto/arquivo.csv'

    dataset = datasource.import_dataset(
        dataset_uri=dataset_uri,
        dataset_name='meudatasetcorreto'
    )

It generates a DHash

If you already connected your data source to MLOps and imported a data set, you can get your data set using DHash:

.. code-block:: python

    dataset = datasource.get_dataset(dataset_hash='D66c8bc440dc4882bfeff40c0dac11641c3583f3aa274293b15ed5db21000b49')


Deleting data source and Data set
---------------------------------

If you want to remove a data source or a data set, you can do as the following example:

.. code-block:: python

    datasource.delete()
