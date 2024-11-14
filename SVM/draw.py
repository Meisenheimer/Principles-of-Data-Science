import os
import numpy as np
from matplotlib import pyplot


def loadFile(filename):
    fp = open(filename, "r", encoding="UTF-8")
    text = fp.readlines()[-4:]
    fp.close()
    train_acc = float(text[0].split(' ')[3][5:-1])
    test_acc = float(text[1].split(' ')[3][5:-1])
    train_auc = float(text[2].split(' ')[3][5:-1])
    test_auc = float(text[3].split(' ')[3][5:-1])
    return train_acc, train_auc, test_acc, test_auc


def draw(filelist, output):
    train_acc_list = []
    train_auc_list = []
    test_acc_list = []
    test_auc_list = []

    for filename in filelist:
        train_acc, train_auc, test_acc, test_auc = loadFile(filename)
        train_acc_list.append(train_acc)
        train_auc_list.append(train_auc)
        test_acc_list.append(test_acc)
        test_auc_list.append(test_auc)

    pyplot.clf()
    pyplot.figure(figsize=(10, 3))
    pyplot.grid()
    x = np.arange(0.0, 2.5 * len(train_acc_list), 2.5)
    pyplot.bar(x + 0.0, train_acc_list, width=0.5, align="edge", color="r", label="train ACC")
    pyplot.bar(x + 0.5, test_acc_list, width=0.5, align="edge", color="g", label="test ACC")
    pyplot.bar(x + 1.0, train_auc_list, width=0.5, align="edge", color="b", label="train AUC")
    pyplot.bar(x + 1.5, test_auc_list, width=0.5, align="edge", color="c", label="test AUC")
    pyplot.ylim([0.7, 1.0])
    pyplot.xticks(ticks=x + 1, labels=[
        r"$N_{node} = 15$" + "\nStatic",
        r"$N_{node} = 25$" + "\nStatic",
        r"$N_{node} = 50$" + "\nStatic",
        r"$N_{node} = 15$" + "\nDynamic",
        r"$N_{node} = 25$" + "\nDynamic",
        r"$N_{node} = 50$" + "\nDynamic"
    ])
    pyplot.legend(loc="lower right")
    pyplot.savefig(output, dpi=720, bbox_inches="tight")


if __name__ == "__main__":
    draw([
        "linear_node=15_size=4800_step=4800_rho=0.100000_num_iter=150.log",
        "linear_node=25_size=4800_step=4800_rho=0.100000_num_iter=150.log",
        "linear_node=50_size=4800_step=4800_rho=0.100000_num_iter=150.log",
        "linear_node=15_size=480_step=180_rho=0.100000_num_iter=150.log",
        "linear_node=25_size=480_step=180_rho=0.100000_num_iter=150.log",
        "linear_node=50_size=480_step=180_rho=0.100000_num_iter=150.log"
    ], "linear_0.1.jpg")
    draw([
        "nu_node=15_size=4800_step=4800_rho=0.100000_num_iter=150.log",
        "nu_node=25_size=4800_step=4800_rho=0.100000_num_iter=150.log",
        "nu_node=50_size=4800_step=4800_rho=0.100000_num_iter=150.log",
        "nu_node=15_size=480_step=180_rho=0.100000_num_iter=150.log",
        "nu_node=25_size=480_step=180_rho=0.100000_num_iter=150.log",
        "nu_node=50_size=480_step=180_rho=0.100000_num_iter=150.log"
    ], "nu_0.1.jpg")
    # draw([
    #     "linear_node=15_size=4800_step=4800_rho=0.010000_num_iter=150.log",
    #     "linear_node=25_size=4800_step=4800_rho=0.010000_num_iter=150.log",
    #     "linear_node=50_size=4800_step=4800_rho=0.010000_num_iter=150.log",
    #     "linear_node=15_size=480_step=180_rho=0.010000_num_iter=150.log",
    #     "linear_node=25_size=480_step=180_rho=0.010000_num_iter=150.log",
    #     "linear_node=50_size=480_step=180_rho=0.010000_num_iter=150.log"
    # ], "linear_0.01.jpg")
    # draw([
    #     "nu_node=15_size=4800_step=4800_rho=0.010000_num_iter=150.log",
    #     "nu_node=25_size=4800_step=4800_rho=0.010000_num_iter=150.log",
    #     "nu_node=50_size=4800_step=4800_rho=0.010000_num_iter=150.log",
    #     "nu_node=15_size=480_step=180_rho=0.010000_num_iter=150.log",
    #     "nu_node=25_size=480_step=180_rho=0.010000_num_iter=150.log",
    #     "nu_node=50_size=480_step=180_rho=0.010000_num_iter=150.log"
    # ], "nu_0.01.jpg")
