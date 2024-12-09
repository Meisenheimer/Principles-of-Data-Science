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
index = []


def makeMatrix(x, args):
    res = np.zeros((args.node, args.node))
    k = 0
    for i in range(args.node):
        for j in range(i + 1, args.node):
            res[i][j] = res[j][i] = x[k]
            k += 1
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--node", type=int, required=True)

    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--rho", type=float, required=True)

    args = parser.parse_args()

    args.input_size = math.floor((TOTAL_TIME - args.size) / args.step) + 1

    args.output = f"./MIR/node={args.node}_size={args.size}_step={args.step}_rho={args.rho}"

    os.makedirs(args.output, exist_ok=True)
    flog = open(os.path.join(args.output, "output.log"), "w")

    id_list = loadIdList()

    x = []
    y = []

    for i in range(args.node):
        for j in range(i + 1, args.node):
            index.append((i, j))

    for id, gender, age in tqdm(id_list):
        fc = loadData(args.node, args.size, args.step, args.rho, id)
        x.append(fc)
        y.append(gender)

    x = np.array(x)
    y = np.array(y)

    mean = np.zeros((args.node * (args.node - 1) // 2))

    for t in tqdm(range(args.input_size)):
        model = LinearDiscriminantAnalysis(n_components=1)
        model.fit(x[:, t, :].reshape(-1, args.node * (args.node - 1) // 2), y)

        tmp = model.scalings_.reshape(-1)
        mean += tmp
        m = makeMatrix(tmp, args)

        # pyplot.clf()
        # pyplot.grid()
        # pyplot.imshow(m)
        # pyplot.colorbar()
        # pyplot.savefig(os.path.join(args.output, f"heatmap_{t}.jpg"), dpi=720, bbox_inches="tight")

        # for i in np.abs(tmp).argsort()[-len(tmp) // 10:]:
        #     print("(%d, %d)" % index[i], file=flog, end=" ")
        # print("", file=flog)

        # for i in range(args.node):
        #     for j in range(args.node):
        #         print("%5.2f" % m[i][j], file=flog, end=" ")
        #     print("", file=flog)
        # print("", file=flog)

    mean /= args.input_size

    tmp = mean
    mean = makeMatrix(tmp, args)

    pyplot.clf()
    pyplot.grid()
    pyplot.imshow(mean)
    pyplot.colorbar()
    pyplot.savefig(os.path.join(args.output, f"heatmap_mean.jpg"), dpi=720, bbox_inches="tight")

    for i in np.abs(tmp).argsort()[-len(tmp) // 10:]:
        print("(%d, %d)" % index[i], file=flog, end=" ")
    print("", file=flog)

    for i in range(args.node):
        for j in range(args.node):
            print("%5.2f" % mean[i][j], file=flog, end=" ")
        print("", file=flog)

    for i in range(args.node):
        print("%d: %5.2f" % (i, sum(np.abs(mean[i, :]))), file=flog)
    print("", file=flog)

    flog.close()
