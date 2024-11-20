import os
import math
import numpy as np
from matplotlib import pyplot
from tqdm import tqdm
import argparse
from scipy.stats import ecdf, kstest

from Config import loadData, loadIdList

TOTAL_TIME = 4800
flog = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--node", type=int, required=True)

    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--rho", type=float, required=True)

    parser.add_argument("--figure", type=bool, default=False)

    parser.add_argument("--normalize", type=bool, default=False)

    args = parser.parse_args()

    args.input_size = math.floor((TOTAL_TIME - args.size) / args.step) + 1

    args.output = f"./Normal/node={args.node}_size={args.size}_step={args.step}_rho={args.rho}{'_normalize' if args.normalize else ''}"

    os.makedirs(args.output, exist_ok=True)
    flog = open(os.path.join(args.output, "output.log"), "w")

    id_list = loadIdList()

    list_m = []
    list_f = []

    pass_ks_m = 0.0
    pass_ks_f = 0.0
    min_ks_m = 1.0
    min_ks_f = 1.0
    max_ks_m = 0.0
    max_ks_f = 0.0
    avg_ks_m = 0.0
    avg_ks_f = 0.0

    for id, gender, age in tqdm(id_list):
        fc = loadData(args.node, args.size, args.step, args.rho, id)
        if (gender == 1):
            list_m.append(fc)
        elif (gender == 0):
            list_f.append(fc)

    list_m = np.array(list_m)
    list_f = np.array(list_f)

    for t in tqdm(range(args.input_size)):
        ks_m = np.zeros((args.node * (args.node - 1) // 2, ))
        ks_f = np.zeros((args.node * (args.node - 1) // 2, ))
        for i in range(args.node * (args.node - 1) // 2):
            m = list_m[:, t, i].reshape(-1)
            f = list_f[:, t, i].reshape(-1)

            if (args.normalize):
                m = (m - m.mean()) / m.std()
                f = (f - f.mean()) / f.std()

            ks_m[i] = kstest(m, cdf="norm").pvalue
            ks_f[i] = kstest(f, cdf="norm").pvalue

            if (args.figure and i < args.node and t < 1):
                ecdf_m = ecdf(m).cdf
                ecdf_f = ecdf(f).cdf

                pyplot.clf()
                pyplot.grid()
                pyplot.plot(ecdf_m.quantiles, ecdf_m.probabilities, color="blue", label=f"Male")
                pyplot.plot(ecdf_f.quantiles, ecdf_f.probabilities, color="red", label=f"Female")
                pyplot.legend()
                pyplot.savefig(os.path.join(args.output, f"cdf_{t}_{i}.jpg"), dpi=720, bbox_inches="tight")

        tmp_ks_m = np.sum(ks_m >= 0.05) / (args.node * (args.node - 1) / 2)
        tmp_ks_f = np.sum(ks_f >= 0.05) / (args.node * (args.node - 1) / 2)
        pass_ks_m += tmp_ks_m
        pass_ks_f += tmp_ks_f
        min_ks_m = min(min_ks_m, ks_m.min())
        min_ks_f = min(min_ks_f, ks_f.min())
        max_ks_m = max(max_ks_m, ks_m.max())
        max_ks_f = max(max_ks_f, ks_f.max())
        avg_ks_m += ks_m.mean()
        avg_ks_f += ks_f.mean()
        print(f"t = {t:.4f}, min(ks_m) = {ks_m.min():.4f}, max(ks_m) = {ks_m.max():.4f}, cnt(pvalue >= 0.05) = {np.sum(ks_m >= 0.05):.4f}({tmp_ks_m:.4f}%)", file=flog)
        print(f"t = {t:.4f}, min(ks_f) = {ks_f.min():.4f}, max(ks_f) = {ks_f.max():.4f}, cnt(pvalue >= 0.05) = {np.sum(ks_f >= 0.05):.4f}({tmp_ks_f:.4f}%)", file=flog)

        if (args.figure):
            ks_m[ks_m < 1e-7] = 0.0
            ks_f[ks_f < 1e-7] = 0.0

            pyplot.clf()
            pyplot.grid()
            pyplot.xlim([0.0, 1.0])
            pyplot.hist(ks_m.reshape(-1), bins=20, range=(0.0, 1.0))
            pyplot.vlines([0.05], [0], [20], linestyles="dotted")
            pyplot.savefig(os.path.join(args.output, f"hist_ks_m_{t}.jpg"), dpi=720, bbox_inches="tight")

            pyplot.clf()
            pyplot.grid()
            pyplot.xlim([0.0, 1.0])
            pyplot.hist(ks_f.reshape(-1), bins=20, range=(0.0, 1.0))
            pyplot.vlines([0.05], [0], [20], linestyles="dotted")
            pyplot.savefig(os.path.join(args.output, f"hist_ks_f_{t}.jpg"), dpi=720, bbox_inches="tight")

            ks_m = np.clip(ks_m, 0.0, 0.05)
            ks_f = np.clip(ks_f, 0.0, 0.05)

    print(f"min_ks_m = {min_ks_m:.4f}, min_ks_f = {min_ks_f:.4f}", file=flog)
    print(f"max_ks_m = {max_ks_m:.4f}, max_ks_f = {max_ks_f:.4f}", file=flog)
    print(f"pass_ks_m = {pass_ks_m / args.input_size:.4f}, pass_ks_f = {pass_ks_f / args.input_size:.4f}", file=flog)

    data_type = "Static" if args.input_size == 1 else "Dynamic"
    print(f"Male & {data_type} & ${args.node}$ &  ${min_ks_m:.4f}$ & ${max_ks_m:.4f}$ & ${(avg_ks_m / args.input_size):.4f}$ & ${(pass_ks_m / args.input_size):.4f}$", file=flog)
    print(f"Female & {data_type} & ${args.node}$ &  ${min_ks_f:.4f}$ & ${max_ks_f:.4f}$ & ${(avg_ks_f / args.input_size):.4f}$ & ${(pass_ks_f / args.input_size):.4f}$", file=flog)

    flog.close()
