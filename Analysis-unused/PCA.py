import os
import math
import argparse
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot
from sklearn.decomposition import PCA

from Config import loadData, loadIdList


TOTAL_TIME = 4800
flog = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--node", type=int, required=True)

    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--rho", type=float, required=True)

    args = parser.parse_args()

    args.input_size = math.floor((TOTAL_TIME - args.size) / args.step) + 1

    args.output = f"./PCA/node={args.node}_size={args.size}_step={args.step}_rho={args.rho}"

    os.makedirs(args.output, exist_ok=True)
    flog = open(os.path.join(args.output, "output.log"), "w")

    id_list = loadIdList()

    x = []
    y = []
    dummy = []
    color = []
    label = []
    score = []

    for id, gender, age in tqdm(id_list):
        fc = loadData(args.node, args.size, args.step, args.rho, id)
        x.append(fc)
        y.append(gender)
        color.append("red" if gender == 0 else "blue")
        label.append("Female" if gender == 0 else "Male")

    x = np.array(x)
    y = np.array(y)

    model = PCA(n_components=2)
    pca = model.fit_transform(x.reshape(len(id_list), -1), y)

    pyplot.clf()
    pyplot.grid()
    pyplot.scatter(pca[:, 0], pca[:, 1], color=color, label=label)
    # pyplot.legend()
    pyplot.savefig(os.path.join(args.output, f"pca.jpg"), dpi=720, bbox_inches="tight")

    for t in tqdm(range(args.input_size)):
        model = PCA(n_components=2)
        pca = model.fit_transform(x[:, t, :].reshape(-1, args.node * (args.node - 1) // 2), y)

        pyplot.clf()
        pyplot.grid()
        pyplot.scatter(pca[:, 0], pca[:, 1], color=color, label=label)
        # pyplot.legend()
        pyplot.savefig(os.path.join(args.output, f"pca_{t}.jpg"), dpi=720, bbox_inches="tight")

    flog.close()
