import random
import matplotlib.pyplot as plt
import numpy as np
import os


def calculate_P_R(TP: int, TN: int, FP: int, FN: int) -> (float, float):
    """
    :param TP: true positive: int
    :param TN:  true negative: int
    :param FP: false positive: int
    :param FN: false negative: int
    :return: precision, recall: float
    """

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    return precision, recall


def calculate_F_beta(precision: float, recall: float, beta: int = 2):
    """
    :param precision: precision: float
    :param recall: recall: float
    :param beta: beta: int
    :return: F_beta: float
    """
    return (1 + beta ** 2) * precision * recall / (((beta ** 2) * precision) + recall)


def plot_F_beta():
    TP = np.arange(1, 500, 20)
    TN = np.arange(1, 500, 20)

    TP_mesh, TN_mesh = np.meshgrid(TP, TN)
    FN_mesh = 500 - TP_mesh
    FP_mesh = 500 - TN_mesh
    precision, recall = calculate_P_R(TP_mesh, TN_mesh, FP_mesh, FN_mesh)
    F_beta = calculate_F_beta(precision, recall)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_wireframe(precision, recall, F_beta)

    ax.set_xlabel("TP")
    ax.set_ylabel("TN")
    ax.set_zlabel("F_beta")

    plt.show()


if __name__ == "__main__":
    plot_F_beta()
