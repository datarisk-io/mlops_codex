import os
import yaml
import json
from loguru import logger
from typing import Optional
from datetime import datetime

from neomaril_codex.exceptions import *
from neomaril_codex.training import *
from neomaril_codex.model import *

class NeomarilPipeline:
    def __init__(self, password:str, group:str, enviroment:str='staging', python_version:float=3.9) -> None:
        self.__credentials = os.getenv('NEOMARIL_TOKEN') if os.getenv('NEOMARIL_TOKEN') else password
        self.enviroment = os.getenv('NEOMARIL_ENVIROMENT') if os.getenv('NEOMARIL_ENVIROMENT') else enviroment
        self.group = group
        self.python_version = python_version
        self.train_config = None
        self.deploy_config = None
        self.monitoring_config = None

    def register_train_config(self, **kwargs):
        self.train_config = kwargs

    def register_deploy_config(self, **kwargs):
        self.deploy_config = kwargs

    def register_monitoring_config(self, **kwargs):
        self.monitoring_config = kwargs

    @staticmethod
    def from_config_file(path):
        with open(path, 'r') as stream:
            try:
                conf=yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        load_dotenv()

        token = os.getenv("NEOMARIL_TOKEN")
        if not token:
            raise PipelineError("When using a config file the enviroment variable NEOMARIL_TOKEN must be defined")

        pipeline = NeomarilPipeline(token, conf['group'], enviroment=conf['enviroment'], python_version=conf['python_version'])

        if 'training' in conf.keys():
            pipeline.register_train_config(**conf['training'])

        if 'deploy' in conf.keys():
            pipeline.register_deploy_config(**conf['deploy'])

        if 'monitoring' in conf.keys():
            pipeline.register_monitoring_config(**conf['monitoring'])
              
        return pipeline

    def run_training(self):
        logger.info('Running training')
        client = NeomarilTrainingClient(self.__credentials, enviroment=self.enviroment)
        client.create_group(self.group, self.group)

        conf = self.train_config

        training = client.create_training_experiment(conf['experiment_name'], conf['model_type'], conf['training_type'], group=self.group)

        
        PATH = conf['directory']
        run_name = conf.get('run_name', 'Pipeline run '+str(datetime.now()))
        extra_files = conf.get('extra')

        if conf['training_type'] == 'Custom':
            run = training.run_training(run_name, os.path.join(PATH, conf['data']), source_file=os.path.join(PATH, conf['source']),
                                        requirements_file=os.path.join(PATH, conf['packages']), training_reference=conf['train_function'],
                                        extra_files=[os.path.join(PATH,e) for e in extra_files] if extra_files else None,
                                        python_version=str(self.python_version), wait_complete=True)

        elif conf['training_type'] == 'AutoML':
            run = training.run_training(run_name, os.path.join(PATH, conf['data']), os.path.join(PATH, conf['config']), wait_complete=True)

        status = run.get_status()
        if status['Status'] == "Succeeded":
            logger.info('Model training finished')
            return training.training_id, run.exec_id
        else:
            raise TrainingError('Training failed: '+status['Message'])

    def run_deploy(self, training_id:Optional[str]=None):
        conf = self.deploy_config
        PATH = conf['directory']
        extra_files = conf.get('extra')

        if training_id:
            logger.info('Deploying scorer from training')
            training_run = NeomarilTrainingExecution(training_id[0], self.group, training_id[1], password=self.__credentials, 
                                                     enviroment=self.enviroment)

            model_name = conf.get('name', training_run.execution_data.get('ExperimentName', ''))

            if training_run.execution_data['TrainingType'] == 'Custom':
                model = training_run.promote_model(model_name, model_reference=conf['score_function'], 
                                                    source_file=os.path.join(PATH, conf['source']),
                                                    extra_files=[os.path.join(PATH,e) for e in extra_files] if extra_files else None,
                                                    env=os.path.join(PATH, conf['env']) if conf.get('env') else None,
                                                    schema=os.path.join(PATH, conf['schema']) if conf.get('schema') else None,
                                                    operation=conf['operation'])

            elif training_run.execution_data['TrainingType'] == 'AutoML':
                model = training_run.promote_model(model_name, operation=conf['operation'])

        else:
            logger.info('Deploying scorer')
            client = NeomarilModelClient(password=self.__credentials, enviroment=self.enviroment)
            client.create_group(self.group, self.group)
            
            model = client.create_model(conf.get('name'), conf['score_function'], os.path.join(PATH, conf['source']), 
                                        os.path.join(PATH, conf['model']), os.path.join(PATH, conf['packages']),
                                        extra_files=[os.path.join(PATH,e) for e in extra_files] if extra_files else None,
                                        env=os.path.join(PATH, conf['env']) if conf.get('env') else None,
                                        schema=os.path.join(PATH, conf['schema']) if conf.get('schema') else None,
                                        operation=conf['operation'], input_type=conf['input_type'], group=self.group)

        while model.status == 'Building':
            model.wait_ready()

        if model.status == 'Deployed':
            logger.info('Model deployement finished')
            return model.model_id

        else:
            raise ModelError("Model deployement failed: "+ model.get_logs(routine='Host')[0])



    def run_monitoring(self, training_exec_id:Optional[str]=None, model_id:Optional[str]=None):
        logger.info('Configuring monitoring')

        conf = self.monitoring_config
        PATH = conf['directory']

        if training_exec_id:
            with open(os.path.join(PATH, conf['config']), 'r+') as f:
                conf_dict = json.load(f)
                f.seek(0)
                conf_dict['TrainData']['NeomarilTrainingExecution'] = training_exec_id
                json.dump(conf_dict, f)
                f.truncate()

        model = NeomarilModel(self.__credentials, model_id, group=self.group, group_token=os.getenv('NEOMARIL_GROUP_TOKEN'),
                                enviroment=self.enviroment)

        model.register_monitoring(conf['preprocess_function'], conf['shap_function'], 
                                    configuration_file=os.path.join(PATH, conf['config']),
                                    preprocess_file=os.path.join(PATH, conf['preprocess']),
                                    requirements_file=(os.path.join(PATH, conf['packages']) if conf.get('packages') else None))
                           
    
    def start(self):
        if (not self.train_config) and (not self.deploy_config) and (not self.monitoring_config):
            raise PipelineError("Cannot start pipeline without configuration")

        if self.train_config:
            training_id = self.run_training()
        else:
            training_id = None

        if self.deploy_config:
            model_id = self.run_deploy(training_id=training_id)
        else:
            model_id = None

        if self.monitoring_config:
            self.run_monitoring(training_exec_id=(training_id[1] if training_id else None), model_id=model_id)
        