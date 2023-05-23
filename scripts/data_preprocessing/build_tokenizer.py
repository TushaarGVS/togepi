import os
from argparse import ArgumentParser

import datasets
import yaml

from togepi.data_processors.tokenizers.tokenizer import TogepiTokenizer
from togepi.utils.utils import set_seed


def main(config_path, hf_dataset_path, path_to_store_hf_tokenizer):
    set_seed(seed=42)

    with open(config_path, 'r') as fp:
        config = yaml.safe_load(fp)

    hf_dataset = datasets.load_from_disk(dataset_path=hf_dataset_path)
    tokenizer = TogepiTokenizer(**config['tokenizer']['args'])
    tokenizer.train_tokenizer(hf_dataset['train'])
    tokenizer.save(path_to_store_hf_tokenizer)


if __name__ == '__main__':
    parser = ArgumentParser(description='build tokenizer using a huggingface dataset')
    parser.add_argument('--config_path', type=str, help='path to config file')
    parser.add_argument('--hf_dataset_path', type=str, help='path to the huggingface dataset', default=os.getcwd())
    parser.add_argument('--path_to_store_hf_tokenizer', type=str, help='path to store the tokenizer',
                        default=os.getcwd())

    args = parser.parse_args()

    main(config_path=args.config_path, hf_dataset_path=args.hf_dataset_path,
         path_to_store_hf_tokenizer=args.path_to_store_hf_tokenizer)
