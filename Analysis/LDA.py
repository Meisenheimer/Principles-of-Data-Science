import os
import math
import argparse
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

    args.output = f"./LDA/node={args.node}_size={args.size}_step={args.step}_rho={args.rho}"

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

    model = LinearDiscriminantAnalysis(n_components=1)
    lda = model.fit_transform(x.reshape(len(id_list), -1), y).reshape(-1)

    pyplot.clf()
    pyplot.grid()
    pyplot.hist(lda[y == 0], color="r", label="Female", alpha=0.5, density=True, bins=25)
    pyplot.vlines(np.percentile(lda[y == 0], (25, 50, 75)), ymin=0, ymax=0.5, color="r", linestyles="--")
    pyplot.hist(lda[y == 1], color="b", label="Male", alpha=0.5, density=True, bins=25)
    pyplot.vlines(np.percentile(lda[y == 1], (25, 50, 75)), ymin=0, ymax=0.5, color="b", linestyles="--")
    pyplot.legend()
    pyplot.savefig(os.path.join(args.output, f"hist.jpg"), dpi=720, bbox_inches="tight")

    pyplot.clf()
    pyplot.grid()
    pyplot.boxplot([lda[y == 0], lda[y == 1]])
    pyplot.xticks(ticks=[1, 2], labels=["Female", "Male"])
    pyplot.savefig(os.path.join(args.output, f"box.jpg"), dpi=720, bbox_inches="tight")

    score_total = model.score(x.reshape(len(id_list), -1), y)
    print(score_total, file=flog)

    for t in tqdm(range(args.input_size)):
        model = LinearDiscriminantAnalysis(n_components=1)
        lda = model.fit_transform(x[:, t, :].reshape(-1, args.node * (args.node - 1) // 2), y).reshape(-1)

        pyplot.clf()
        pyplot.grid()
        pyplot.hist(lda[y == 0], color="r", label="Female", alpha=0.5, density=True, bins=25)
        pyplot.vlines(np.percentile(lda[y == 0], (25, 50, 75)), ymin=0, ymax=0.5, color="r", linestyles="--")
        pyplot.hist(lda[y == 1], color="b", label="Male", alpha=0.5, density=True, bins=25)
        pyplot.vlines(np.percentile(lda[y == 1], (25, 50, 75)), ymin=0, ymax=0.5, color="b", linestyles="--")
        pyplot.legend()
        pyplot.savefig(os.path.join(args.output, f"hist_{t}.jpg"), dpi=720, bbox_inches="tight")

        pyplot.clf()
        pyplot.grid()
        pyplot.boxplot([lda[y == 0], lda[y == 1]])
        pyplot.xticks(ticks=[1, 2], labels=["Female", "Male"])
        pyplot.savefig(os.path.join(args.output, f"box_{t}.jpg"), dpi=720, bbox_inches="tight")

        score.append(model.score(x[:, t, :].reshape(-1, args.node * (args.node - 1) // 2), y))
        print(t, score[-1], file=flog)

    pyplot.clf()
    pyplot.grid()
    pyplot.ylim([0.70, 1.0])
    pyplot.plot(range(len(score)), score, ".b-")
    pyplot.hlines([score_total], 0, len(score))
    pyplot.savefig(os.path.join(args.output, f"score.jpg"), dpi=720, bbox_inches="tight")

    flog.close()
