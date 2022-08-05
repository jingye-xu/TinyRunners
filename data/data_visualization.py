import collections
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

data = pd.read_pickle("data.gz")

# parameters
# data length
length = len(data["data"][0])
# how many figures in a column
number = 5
# folder name
folder = "images/"
if not os.path.exists(folder):
    os.mkdir(folder)

# obtain statistics
data_counts = collections.Counter(data["rhythm"])
# print(data_counts)


# generate images
def data_show(dataset: pd.DataFrame, keys: str, number: int, length: int, cate: str):
    """
    :param dataset: filtered dataset
    :param keys: data's key
    :param number: how many figures in a column
    :param length: data length
    :param cate: category
    :return: None
    """
    # x axis array
    x = np.arange(0, length, 1)

    # plan axs
    figs, ax = plt.subplots(number, 1, sharex="col")
    ax[0].set_title(cate)

    # plot
    for i in range(len(ax)):
        ax[i].plot(x, dataset[keys][i])

    # save
    plt.savefig(f"{folder}{cate}.svg", dpi=600)


# enumerate cates and save images
for category in data_counts:
    sub_data = data[data["rhythm"] == category]
    data_show(sub_data.reset_index(drop=True), "data", number, length, category)
