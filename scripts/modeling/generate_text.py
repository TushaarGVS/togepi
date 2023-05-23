import os
from argparse import ArgumentParser

import torch
import yaml

from togepi.data_processors.tokenizers.tokenizer import TogepiTokenizer
from togepi.models.transformer import Transformer
from togepi.utils.utils import set_seed


def main(prompt, config_path, tokenizer_path, pretrained_checkpoint_path, pretrained_model_path=None):
    set_seed(seed=42)

    with open(config_path, 'r') as fp:
        config = yaml.safe_load(fp)

    togepi_tokenizer = TogepiTokenizer.load(tokenizer_path)
    transformer = Transformer(vocab_size=togepi_tokenizer.vocab_size, padding_idx=togepi_tokenizer.pad_token_id,
                              **config['transformer']['args'])

    if pretrained_checkpoint_path is not None:
        checkpoint = torch.load(pretrained_checkpoint_path)
        transformer.load_state_dict(checkpoint['model_state_dict'])
    elif pretrained_model_path is not None:
        transformer.from_pretrained(model_save_path=pretrained_model_path)

    tok_text = TogepiTokenizer.tokenize(pretrained_tokenizer=togepi_tokenizer, text=prompt, max_length=None)
    # ignore the last '<|endoftext|>' token added at the end of the sequence by the tokenizer
    input_ids = torch.tensor(tok_text['input_ids'][:-1]).expand(config['generate']['args']['num_samples'], -1)

    augmented_input_ids = transformer.generate(input_ids=input_ids, **config['generate']['args'])
    for sample_idx in range(config['generate']['args']['num_samples']):
        print(togepi_tokenizer.decode(augmented_input_ids[sample_idx].cpu().squeeze()))
        print('-' * 80)


if __name__ == '__main__':
    parser = ArgumentParser(description='use the trained language model to generate text')
    parser.add_argument('--config_path', type=str, help='path to config file')
    parser.add_argument('--tokenizer_path', type=str, help='path to the pretrained tokenizer', default=os.getcwd())
    parser.add_argument('--pretrained_model_path', type=str, help='path to the pretrained model', default=None)
    parser.add_argument('--pretrained_checkpoint_path', type=str, help='path to the pretrained model checkpoint',
                        default=None)
    parser.add_argument('--prompt', type=str, help='the input prompt (to be completed)')

    args = parser.parse_args()

    main(config_path=args.config_path, tokenizer_path=args.tokenizer_path,
         pretrained_model_path=args.pretrained_model_path, pretrained_checkpoint_path=args.pretrained_checkpoint_path,
         prompt=args.prompt)
