import os
import math
import argparse
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot

from Config import loadData, loadIdList, makeMat

SIZE = 480
STEP = 180
RHO = 0.1
TOTAL_TIME = 4800
N_NODE = 15
INPUT_SIZE = math.floor((TOTAL_TIME - SIZE) / STEP) + 1
flog = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--rho", type=float, required=True)
    args = parser.parse_args()

    RHO = args.rho

    input_size = math.floor((TOTAL_TIME - SIZE) / STEP) + 1
    output = f"./DFC/size={SIZE}_step={STEP}_rho={RHO}/"
    os.makedirs(output, exist_ok=True)
    flog = open(os.path.join(output, "output.log"), "w")

    for node in [15, 25, 50]:
        fc = loadData(node, SIZE, STEP, RHO, args.id)
        figure_output = os.path.join(output, f"node={node}_id={args.id}/")
        os.makedirs(figure_output, exist_ok=True)
        for t in tqdm(range(INPUT_SIZE)):
            pyplot.clf()
            pyplot.grid()
            pyplot.imshow(makeMat(node, fc[t]), vmin=-0.6, vmax=0.6)
            pyplot.colorbar()
            pyplot.savefig(os.path.join(figure_output, "%d.jpg" % t), dpi=720, bbox_inches="tight")

            pyplot.clf()
            pyplot.grid()
            pyplot.imshow(makeMat(node, fc[t] - fc.mean(0)), vmin=-0.3, vmax=0.3)
            pyplot.colorbar()
            pyplot.savefig(os.path.join(figure_output, "c_%d.jpg" % t), dpi=720, bbox_inches="tight")

    flog.close()
