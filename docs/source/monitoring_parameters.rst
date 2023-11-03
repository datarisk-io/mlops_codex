:orphan:

Monitoring configuration
========================


{
    "period": "week", <string, Mandatory> indicates monitoring frequency running. Can be `day`, `week`, `month` or `year`
    "train_data" : {
        "NeomarilTrainingExecution": "1", <string, Mandatory for model trained inside Neomaril> executionId from training promoted at Neomaril
        // "train_date_col": "date", <string, Mandatory if no field "train_date_ref" was inserted> name of the date column for the records
        "train_date_ref": "2022-09-01", <string, Mandatory if no field "train_date_col" was inserted> date the model data was acquired
    }
    "input_cols": ["mean_radius", "mean_texture"] , <list[string]> name of the features columns that the monitoring will run. Need to be the same of the train data in MLFlow and the output of the pre-process function
    "output_cols": ["proba", "pred"] , <list[string]> name of the output columns that the monitoring will run. Need to be the same of the output data in MLFlow and the output of the model scoring function
}