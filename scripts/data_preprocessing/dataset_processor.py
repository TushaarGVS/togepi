import os
from argparse import ArgumentParser

import datasets
import yaml

from togepi.data_processors.utils import remove_empty_text_entries


def main(config_path, path_to_store_hf_dataset):
    with open(config_path, 'r') as fp:
        config = yaml.safe_load(fp)

    hf_dataset = datasets.load_dataset(**config['dataset']['args'])
    if config['remove_empty_text_entries']:
        hf_dataset = remove_empty_text_entries(hf_dataset)
    hf_dataset.save_to_disk(path_to_store_hf_dataset)


if __name__ == '__main__':
    parser = ArgumentParser(description='download and save a huggingface dataset')
    parser.add_argument('--config_path', type=str, help='path to config file')
    parser.add_argument('--path_to_store_hf_dataset', type=str, help='path to store huggingface dataset',
                        default=os.getcwd())

    args = parser.parse_args()

    main(config_path=args.config_path, path_to_store_hf_dataset=args.path_to_store_hf_dataset)
