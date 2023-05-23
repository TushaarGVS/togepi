from torch.utils.data import RandomSampler, DataLoader


def remove_empty_text_entries(hf_dataset, text_colname='text'):
    return hf_dataset.filter(lambda data_inst: data_inst[text_colname] != '')


def get_torch_dataset(tok_hf_dataset):
    tok_hf_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask'])
    return tok_hf_dataset


def get_dataloader(torch_dataset, batch_size=128, shuffle=False, num_samples=None, num_workers=0):
    if num_samples is not None and shuffle:
        replacement = False
        if num_samples > len(torch_dataset):
            replacement = True
        sampler = RandomSampler(torch_dataset, replacement=replacement, num_samples=num_samples)
        return DataLoader(torch_dataset, sampler=sampler, batch_size=batch_size, pin_memory=True,
                          num_workers=num_workers)
    else:
        return DataLoader(torch_dataset, shuffle=shuffle, batch_size=batch_size, pin_memory=True,
                          num_workers=num_workers)
