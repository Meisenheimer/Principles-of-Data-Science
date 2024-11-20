import os
import numpy as np
from matplotlib import pyplot


def loadFile(filename):
    fp = open(filename, "r", encoding="UTF-8")
    text = fp.readlines()
    fp.close()

    train_acc_list = []
    test_acc_list = []
    train_auc_list = []
    test_auc_list = []

    for item in text[1:151]:
        tmp = item.split(',')
        train_acc_list.append(float(tmp[0][10:]))
        test_acc_list.append(float(tmp[1][10:]))
        train_auc_list.append(float(tmp[2][11:]))
        test_auc_list.append(float(tmp[3][10:]))
    return train_acc_list, train_auc_list, test_acc_list, test_auc_list


def drawBox(filelist, output):
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
    pyplot.figure(figsize=(5, 3))
    pyplot.grid()
    pyplot.boxplot(test_auc_list, vert=False)
    pyplot.yticks(ticks=range(1, 7), labels=[r"$N_{node} = 15$ sFC", r"$N_{node} = 25$ sFC", r"$N_{node} = 50$ sFC", r"$N_{node} = 15$ dFC", r"$N_{node} = 25$ dFC", r"$N_{node} = 50$ dFC"])
    pyplot.savefig("auc_" + output, dpi=720, bbox_inches="tight")
    pyplot.close()

    pyplot.clf()
    pyplot.figure(figsize=(5, 3))
    pyplot.grid()
    pyplot.boxplot(test_acc_list, vert=False)
    pyplot.yticks(ticks=range(1, 7), labels=[r"$N_{node} = 15$ sFC", r"$N_{node} = 25$ sFC", r"$N_{node} = 50$ sFC", r"$N_{node} = 15$ dFC", r"$N_{node} = 25$ dFC", r"$N_{node} = 50$ dFC"])
    pyplot.savefig("acc_" + output, dpi=720, bbox_inches="tight")
    pyplot.close()


def draw(filelist, output):
    train_acc_list = []
    train_auc_list = []
    test_acc_list = []
    test_auc_list = []

    for filename in filelist:
        train_acc, train_auc, test_acc, test_auc = loadFile(filename)
        print(filename.replace("_size=480_step=180_rho=0.100000_", " Dynamic ").replace("_size=4800_step=4800_rho=0.100000_", " Static ").replace("num_iter=150.log", "").replace("_", " "), f"train acc min={np.min(train_acc):.4f}", f"train auc min={np.min(train_auc):.4f}", f"test acc min={np.min(test_acc):.4f}", f"test auc min={np.min(test_auc):.4f}")
        print(filename.replace("_size=480_step=180_rho=0.100000_", " Dynamic ").replace("_size=4800_step=4800_rho=0.100000_", " Static ").replace("num_iter=150.log", "").replace("_", " "), f"train acc mean={np.mean(train_acc):.4f}", f"train auc mean={np.mean(train_auc):.4f}", f"test acc mean={np.mean(test_acc):.4f}", f"test auc mean={np.mean(test_auc):.4f}")
        print(filename.replace("_size=480_step=180_rho=0.100000_", " Dynamic ").replace("_size=4800_step=4800_rho=0.100000_", " Static ").replace("num_iter=150.log", "").replace("_", " "), f"train acc max={np.max(train_acc):.4f}", f"train auc max={np.max(train_auc):.4f}", f"test acc max={np.max(test_acc):.4f}", f"test auc max={np.max(test_auc):.4f}")
        train_acc_list.append(np.mean(train_acc))
        train_auc_list.append(np.mean(train_auc))
        test_acc_list.append(np.mean(test_acc))
        test_auc_list.append(np.mean(test_auc))

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
    pyplot.close()


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
    drawBox([
        "linear_node=15_size=4800_step=4800_rho=0.100000_num_iter=150.log",
        "linear_node=25_size=4800_step=4800_rho=0.100000_num_iter=150.log",
        "linear_node=50_size=4800_step=4800_rho=0.100000_num_iter=150.log",
        "linear_node=15_size=480_step=180_rho=0.100000_num_iter=150.log",
        "linear_node=25_size=480_step=180_rho=0.100000_num_iter=150.log",
        "linear_node=50_size=480_step=180_rho=0.100000_num_iter=150.log"
    ], "linear_0.1.jpg")
    drawBox([
        "nu_node=15_size=4800_step=4800_rho=0.100000_num_iter=150.log",
        "nu_node=25_size=4800_step=4800_rho=0.100000_num_iter=150.log",
        "nu_node=50_size=4800_step=4800_rho=0.100000_num_iter=150.log",
        "nu_node=15_size=480_step=180_rho=0.100000_num_iter=150.log",
        "nu_node=25_size=480_step=180_rho=0.100000_num_iter=150.log",
        "nu_node=50_size=480_step=180_rho=0.100000_num_iter=150.log"
    ], "nu_0.1.jpg")
