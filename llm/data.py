import torch
import random


def text_to_tensor(text, ctoi):
    return torch.tensor([ctoi[c] for c in text])


def tensor_to_text(tensor, itoc):
    return ''.join(itoc[i.item()] for i in tensor.flatten())


def load_dataset(text_file, *slices):
    with open(text_file, mode='r') as file:
        text = file.read()

    ctoi = {c: i for (i, c) in enumerate(set(text))}
    itoc = {i: c for (c, i) in ctoi.items()}
    datasets = [text_to_tensor(text[s], ctoi) for s in slices]

    return datasets, ctoi, itoc


def iter_sub_sequences(data, seq_len, batch_size=None):
    """
    Arguments
    ---------
    data:
        [dataset_length,] or [dataset_length, batch_size]
    seq_len:
        int
    batch_size:
        int or None

    Yields
    ------
    src_sub_sequence:
        [seq_len,] or [seq_len, batch_size]
    target_sub_sequence:
        [seq_len,] or [seq_len, batch_size]
    """
    if batch_size is not None:
        data = data[:len(data) - (len(data) % batch_size)].view(batch_size, -1).T

    starting_point = random.randrange(len(data))
    for start in range(starting_point, len(data)-1, seq_len):
        end = min(start+seq_len, len(data)-1)
        yield data[start:end], data[start+1:end+1]

    for start in range(0, starting_point, seq_len):
        end = min(start+seq_len, starting_point)
        yield data[start:end], data[start+1:end+1]
