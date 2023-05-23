from togepi.data_processors.tokenizers.tokenizer import TogepiTokenizer


def batch_tokenize(data_instances, pretrained_tokenizer, max_length=None, text_colname='text'):
    tokenized_data = {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}
    for text in data_instances[text_colname]:
        tok_output = TogepiTokenizer.tokenize(pretrained_tokenizer=pretrained_tokenizer, text=text,
                                              max_length=max_length)

        tokenized_data['input_ids'].append(tok_output['input_ids'].squeeze())
        tokenized_data['token_type_ids'].append(tok_output['token_type_ids'].squeeze())
        tokenized_data['attention_mask'].append(tok_output['attention_mask'].squeeze())
    return tokenized_data
