import os
import math
import argparse
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot

from Config import loadData, loadIdList

SIZE = 480
STEP = 180
RHO = 0.1
TOTAL_TIME = 4800
N_NODE = 40
flog = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, required=True)
    parser.add_argument("--rho", type=float, required=True)
    args = parser.parse_args()

    RHO = args.rho

    input_size = math.floor((TOTAL_TIME - SIZE) / STEP) + 1
    output = f"./Dynamic/size={SIZE}_step={STEP}_rho={RHO}/"
    os.makedirs(output, exist_ok=True)
    flog = open(os.path.join(output, "output.log"), "w")

    id_list = loadIdList()

    var = {}
    for node in [15, 25, 50]:
        var[node] = []
        for id, gender, age in tqdm(id_list):
            fc = loadData(node, SIZE, STEP, RHO, id)
            tmp = []
            for i in range(node * (node - 1) // 2):
                tmp.append(np.var(fc[:, i]))
            var[node].append(np.var(tmp))

            if (args.num > 0):
                args.num -= 1
                figure_output = os.path.join(output, f"node={node}_id={id}/")
                os.makedirs(figure_output, exist_ok=True)
                for i in range(0, N_NODE, 4):
                    pyplot.clf()
                    pyplot.grid()
                    pyplot.ylim([-0.6, 0.6])
                    pyplot.hlines([np.mean(fc[:, i])], 0, TOTAL_TIME, colors="r", linestyles="--", alpha=0.5)
                    pyplot.hlines([np.mean(fc[:, i + 1])], 0, TOTAL_TIME, colors="g", linestyles="--", alpha=0.5)
                    pyplot.hlines([np.mean(fc[:, i + 2])], 0, TOTAL_TIME, colors="b", linestyles="--", alpha=0.5)
                    pyplot.hlines([np.mean(fc[:, i + 3])], 0, TOTAL_TIME, colors="c", linestyles="--", alpha=0.5)
                    pyplot.plot(range(SIZE // 2, TOTAL_TIME - SIZE // 2 + 1, STEP), fc[:, i], ".r-", alpha=0.5)
                    pyplot.plot(range(SIZE // 2, TOTAL_TIME - SIZE // 2 + 1, STEP), fc[:, i + 1], ".g-", alpha=0.5)
                    pyplot.plot(range(SIZE // 2, TOTAL_TIME - SIZE // 2 + 1, STEP), fc[:, i + 2], ".b-", alpha=0.5)
                    pyplot.plot(range(SIZE // 2, TOTAL_TIME - SIZE // 2 + 1, STEP), fc[:, i + 3], ".c-", alpha=0.5)
                    pyplot.savefig(os.path.join(figure_output, "%d-%d.jpg" % (i, i + 3)), dpi=720, bbox_inches="tight")
    print(np.min(var[15]), np.mean(var[15]), np.max(var[15]), file=flog)
    print(var[15], file=flog)
    print(np.min(var[25]), np.mean(var[25]), np.max(var[25]), file=flog)
    print(var[25], file=flog)
    print(np.min(var[50]), np.mean(var[50]), np.max(var[50]), file=flog)
    print(var[50], file=flog)

    flog.close()
