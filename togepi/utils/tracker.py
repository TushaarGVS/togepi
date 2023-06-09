import json
import logging
import os
import pickle

import jsonlines
import wandb
from rich.console import Console
from rich.logging import RichHandler


class Tracker(object):
    def __init__(self, config, base_path_to_store_results, experiment_name, project_name='togepi',
                 entity_name='cornell-llms', log_to_wandb=True, resume_wandb_logging=False, log_level=logging.DEBUG):
        super().__init__()

        self.base_path_to_store_results = base_path_to_store_results
        self.config = config
        self.experiment_name = experiment_name
        self.project_name = project_name
        self.entity_name = entity_name
        self.log_to_wandb = log_to_wandb
        self.resume_wandb_logging = resume_wandb_logging
        self.log_level = log_level

        self._setup()

    def _setup(self):
        self._run_path = os.path.join(self.base_path_to_store_results, self.project_name, self.experiment_name)
        self._checkpoints_path = os.path.join(self._run_path, 'checkpoints')
        os.makedirs(self._run_path, exist_ok=True)
        os.makedirs(self._checkpoints_path, exist_ok=True)

        config_path = os.path.join(self._run_path, 'config.json')
        with open(config_path, 'w') as fp:
            json.dump(self.config, fp)

        log_path = os.path.join(self._run_path, 'log.txt')
        logging.basicConfig(level=self.log_level, format="%(asctime)s [%(levelname)s] %(message)s",
                            handlers=[logging.FileHandler(log_path), RichHandler(console=Console(quiet=False))])

        if self.log_to_wandb:
            if self.resume_wandb_logging:
                self._load_run_id()
                self._wandb_run = wandb.init(entity=self.entity_name, project=self.project_name,
                                             name=self.experiment_name, config=self.config, id=self.run_id,
                                             resume='must')
            else:
                self.run_id = wandb.util.generate_id()
                self._save_run_id()
                self._wandb_run = wandb.init(entity=self.entity_name, project=self.project_name,
                                             name=self.experiment_name, config=self.config, id=self.run_id)

    def _save_run_id(self):
        wandb_info_path = os.path.join(self._run_path, 'wandb_info.pkl')
        with open(wandb_info_path, 'wb') as f:
            pickle.dump(self.run_id, f)

    def _load_run_id(self):
        wandb_info_path = os.path.join(self._run_path, 'wandb_info.pkl')
        with open(wandb_info_path, 'rb') as f:
            self.run_id = pickle.load(f)

    def log_metrics(self, epoch_or_step, split_name, metrics, epoch_or_step_id='epoch', log_to_console=True):
        splitwise_metrics_file = os.path.join(self._run_path, f'{split_name}_split_metrics.jsonl')
        metrics_ = {epoch_or_step_id: epoch_or_step, 'metrics': metrics}
        with jsonlines.open(splitwise_metrics_file, 'a') as fp:
            fp.write(metrics_)

        if self.log_to_wandb:
            metrics_ = {f'{split_name}/{epoch_or_step_id}/{metric_key}': value for metric_key, value in metrics.items()}
            metrics_[epoch_or_step_id] = epoch_or_step
            wandb.log(metrics_)

        if log_to_console:
            logging.info(f'{split_name} metrics: {metrics_}')

    def save_model(self, model):
        model_path = os.path.join(self._run_path, 'model.pt')
        model.save_pretrained(model_path)

    def save_checkpoint(self, trainer, epoch):
        checkpoint_path = os.path.join(self._checkpoints_path, f'checkpoint_{epoch}.pt')
        trainer.save_checkpoint(checkpoint_path)

    def done(self):
        if self.log_to_wandb:
            wandb.finish()
