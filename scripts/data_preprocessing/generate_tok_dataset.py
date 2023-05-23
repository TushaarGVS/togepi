import os
from argparse import ArgumentParser

import datasets
import yaml

from togepi.data_processors.tokenizers.tokenizer import TogepiTokenizer
from togepi.data_processors.tokenizers.utils import batch_tokenize


def main(config_path, path_to_store_tokenized_hf_dataset, tokenizer_path, hf_dataset_path):
    with open(config_path, 'r') as fp:
        config = yaml.safe_load(fp)

    togepi_tokenizer = TogepiTokenizer.load(tokenizer_path)
    hf_dataset = datasets.load_from_disk(dataset_path=hf_dataset_path)
    tokenize_helper = lambda data_instance: batch_tokenize(data_instance, pretrained_tokenizer=togepi_tokenizer,
                                                           **config['tokenize_data']['args'])
    tokenized_data = hf_dataset.map(tokenize_helper, batched=True)
    tokenized_data.save_to_disk(path_to_store_tokenized_hf_dataset)


if __name__ == '__main__':
    parser = ArgumentParser(description='generate training dataset (flat conversational corpus)')
    parser.add_argument('--config_path', type=str, help='path to config file')
    parser.add_argument('--path_to_store_tokenized_hf_dataset', type=str, help='path to store huggingface dataset',
                        default=os.getcwd())
    parser.add_argument('--tokenizer_path', type=str, help='path to load the tokenizer from', default=None)
    parser.add_argument('--hf_dataset_path', type=str, help='path to load the huggingface dataset from',
                        default=None)

    args = parser.parse_args()

    main(config_path=args.config_path, path_to_store_tokenized_hf_dataset=args.path_to_store_tokenized_hf_dataset,
         tokenizer_path=args.tokenizer_path, hf_dataset_path=args.hf_dataset_path)
