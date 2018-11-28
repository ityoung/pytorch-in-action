import torch
import numpy as np
import os
import codecs

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)

def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)

def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def prepare_dataset():
    training_set = (
        read_image_file(os.path.join('../data', 'raw', 'train-images-idx3-ubyte')),
        read_label_file(os.path.join('../data', 'raw', 'train-labels-idx1-ubyte'))
    )
    test_set = (
        read_image_file(os.path.join('../data', 'raw', 't10k-images-idx3-ubyte')),
        read_label_file(os.path.join('../data', 'raw', 't10k-labels-idx1-ubyte'))
    )
    with open(os.path.join('../data', 'processed', 'training.pt'), 'wb') as f:
        torch.save(training_set, f)
    with open(os.path.join('../data', 'processed', 'test.pt'), 'wb') as f:
        torch.save(test_set, f)

    print('Done!')


if __name__ == '__main__':
    prepare_dataset()