from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler


def get_torch_dataset(tokenized_dataset, is_labeled_data=False):
    if is_labeled_data:
        tokenized_dataset.set_format(type='torch',
                                     columns=['input_ids', 'position_ids', 'relative_position_ids', 'token_type_ids',
                                              'attention_mask', 'labels'])
    else:
        tokenized_dataset.set_format(type='torch',
                                     columns=['input_ids', 'position_ids', 'relative_position_ids', 'token_type_ids',
                                              'attention_mask'])

    return tokenized_dataset


def get_dataloader(torch_dataset, batch_size=128, shuffle=False, num_samples=None, num_workers=0):
    if num_samples is not None and shuffle is True:
        replacement = False
        if num_samples > len(torch_dataset):
            replacement = True
        sampler = RandomSampler(torch_dataset, replacement=replacement, num_samples=num_samples)
        return DataLoader(torch_dataset, sampler=sampler, batch_size=batch_size, pin_memory=True,
                          num_workers=num_workers)
    return DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers)
