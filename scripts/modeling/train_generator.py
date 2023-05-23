import os
from argparse import ArgumentParser

import datasets
import torch
import torch.nn as nn
import yaml
from torch.optim import Adam

from togepi.data_processors.tokenizers.tokenizer import TogepiTokenizer
from togepi.models.transformer import Transformer
from togepi.trainers.trainer import Trainer
from togepi.utils.tracker import Tracker
from togepi.utils.utils import set_seed


def main(config_path, base_path_to_store_results, tokenizer_path, tokenized_hf_dataset_path, pretrained_model_path=None,
         pretrained_checkpoint_path=None, experiment_name='experiment', project_name='togepi', entity_name='',
         log_to_wandb=True):
    set_seed(seed=42)

    with open(config_path, 'r') as fp:
        config = yaml.safe_load(fp)

    tracker = Tracker(config=config, base_path_to_store_results=base_path_to_store_results,
                      experiment_name=experiment_name, project_name=project_name, entity_name=entity_name,
                      log_to_wandb=log_to_wandb)

    device = config['general']['device']
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    togepi_uncased_tokenizer = TogepiTokenizer.load(tokenizer_path)
    tok_hf_dataset = datasets.load_from_disk(dataset_path=tokenized_hf_dataset_path)
    tok_val_data, tok_test_data = None, None
    if 'val' in tok_hf_dataset:
        tok_val_data = tok_hf_dataset['val']
    if 'test' in tok_hf_dataset:
        tok_test_data = tok_hf_dataset['test']

    transformer = Transformer(vocab_size=togepi_uncased_tokenizer.vocab_size,
                              padding_idx=togepi_uncased_tokenizer.pad_token_id, **config['transformer']['args'])
    optim = Adam(transformer.get_trainable_params(), **config['optim']['args'])
    trainer = Trainer(model=transformer, optim=optim, tracker=tracker, tok_train_data=tok_hf_dataset['train'],
                      tok_val_data=tok_val_data, loss_fn=nn.CrossEntropyLoss, device=device,
                      **config['trainer']['args'])

    if pretrained_checkpoint_path is not None:
        trainer.load_from_checkpoint(checkpoint_path=pretrained_checkpoint_path)
    elif pretrained_model_path is not None:
        transformer.from_pretrained(model_save_path=pretrained_model_path)

    trainer.train_and_eval(**config['train_and_eval']['args']['generator'])

    tracker.done()


if __name__ == '__main__':
    parser = ArgumentParser(description='train language model to generate controlled text')
    parser.add_argument('--config_path', type=str, help='path to config file')
    parser.add_argument('--base_path_to_store_results', type=str, help='base path to store results',
                        default=os.getcwd())
    parser.add_argument('--tokenizer_path', type=str, help='path to the pretrained tokenizer', default=os.getcwd())
    parser.add_argument('--tokenized_hf_dataset_path', type=str, help='path to the huggingface dataset',
                        default=os.getcwd())
    parser.add_argument('--pretrained_model_path', type=str, help='path to the pretrained model', default=None)
    parser.add_argument('--pretrained_checkpoint_path', type=str, help='path to the pretrained model checkpoint',
                        default=None)
    parser.add_argument('--experiment_name', type=str, help='wandb experiment name', default='togepi_experiment')
    parser.add_argument('--project_name', type=str, help='wandb project name', default='togepi')
    parser.add_argument('--entity_name', type=str, help='wandb entity name', default=None)
    parser.add_argument('--log_to_wandb', action='store_true', help='whether to use wandb logging')

    args = parser.parse_args()

    main(config_path=args.config_path, base_path_to_store_results=args.base_path_to_store_results,
         tokenizer_path=args.tokenizer_path, tokenized_hf_dataset_path=args.tokenized_hf_dataset_path,
         pretrained_model_path=args.pretrained_model_path, pretrained_checkpoint_path=args.pretrained_checkpoint_path,
         experiment_name=args.experiment_name, project_name=args.project_name, entity_name=args.entity_name,
         log_to_wandb=args.log_to_wandb)
