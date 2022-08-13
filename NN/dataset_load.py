from torch.utils.data import Dataset
import pandas as pd
import os
from torch import tensor


eight_dict = {
        "AFb": 0,
        "AFt": 1,
        "SR": 2,
        "SVT": 3,
        "VFb": 4,
        "VFt": 5,
        "VPD": 6,
        "VT": 7,
    }


two_dict = {
        "AFb": 0,
        "AFt": 0,
        "SR": 0,
        "SVT": 0,
        "VFb": 1,
        "VFt": 1,
        "VPD": 0,
        "VT": 1,
    }


def eight_classes(y):
    return eight_dict[y]


def two_classes(y):
    return two_dict[y]


class DatasetTiny(Dataset):
    def __init__(self, data_dir="../data", filename="data.gz", transform=None, label_decode=two_classes):
        """
        :param data_dir: data directory
        :param filename: data name
        :param transform: transform methods
        :param label_decode: label decode function
        """

        self.data_dir = data_dir
        self.data_filename = filename
        self.transform = transform
        self.label_decode = label_decode

        # read data
        data_path = os.path.join(self.data_dir, self.data_filename)
        self.data = pd.read_pickle(data_path)

        self.len = self.data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        # read data sequence
        x = tensor(self.data["data"].iloc[idx])
        y = self.data["rhythm"].iloc[idx]
        # reshape data to 1,1250,1 = 1 channel, 1250*1
        x = x.reshape(1, len(x), 1)

        # decode labels
        y = self.label_decode(y)

        # apply transforms
        if self.transform:
            x = self.transform(x)

        return x, y


if __name__ == "__main__":

    # define label decode function: eight_classes or two_classes
    dataset = DatasetTiny(label_decode=eight_classes)
    for i, (x, y) in enumerate(dataset):
        # x has a shape of (1,1250,1), y is a scalar
        print(x, x.shape, y)
        break


# sample output
'''
tensor([[[-0.0017],
         [-0.0958],
         [-0.1647],
         ...,
         [-0.0205],
         [-0.0017],
         [-0.0307]]]) torch.Size([1, 1250, 1]) 0
'''
