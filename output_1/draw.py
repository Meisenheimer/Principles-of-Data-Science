import os
import numpy as np
from matplotlib import pyplot


def loadFile(filename):
    fp = open(filename, "r")
    text = fp.readlines()
    fp.close()
    data = []
    for line in text:
        data.append([float(item) for item in line.split(' ')])
    return np.array(data)


def loadLog(dir):
    train_acc = loadFile(os.path.join(dir, "train_acc.log"))[:, -1]
    test_acc = loadFile(os.path.join(dir, "test_acc.log"))[:, -1]
    train_auc = loadFile(os.path.join(dir, "train_auc.log"))[:, -1]
    test_auc = loadFile(os.path.join(dir, "test_auc.log"))[:, -1]
    return train_acc, test_acc, train_auc, test_auc


dirlist = os.listdir("./")
for dir in dirlist:
    if (not os.path.isdir(dir)):
        continue
    train_acc, test_acc, train_auc, test_auc = loadLog(dir)
    fix = dir.replace("_size=480_step=180_rho=0.100000_", " Dynamic ").replace("_size=4800_step=4800_rho=0.100000_", " Static ").replace("_epochs=64_num_iter=150_", " ").replace("_", " ")
    print(f"{fix}, train_acc min={train_acc.min():.4f}, train_auc min={train_auc.min():.4f}, test_acc min={test_acc.min():.4f}, test_auc min={test_auc.min():.4f}")
    print(f"{fix}, train_acc mean={train_acc.mean():.4f}, train_auc mean={train_auc.mean():.4f}, test_acc mean={test_acc.mean():.4f}, test_auc mean={test_auc.mean():.4f}")
    print(f"{fix}, train_acc max={train_acc.max():.4f}, train_auc max={train_auc.max():.4f}, test_acc max={test_acc.max():.4f}, test_auc max={test_auc.max():.4f}")


for channel in [1, 2, 4]:
    for dropout in [0.0, 0.1]:
        train_acc_list = []
        test_acc_list = []
        train_auc_list = []
        test_auc_list = []
        for size, step in [(4800, 4800), (480, 180)]:
            for node in [15, 25, 50]:
                train_acc, test_acc, train_auc, test_auc = loadLog(f"conv2d_node={node}_size={size}_step={step}_rho=0.100000_dropout={dropout}_epochs=64_num_iter=150_lr=0.001_channel={channel}")
                train_acc_list.append(train_acc.mean())
                test_acc_list.append(test_acc.mean())
                train_auc_list.append(train_auc.mean())
                test_auc_list.append(test_auc.mean())

        for size, step in [(480, 180)]:
            for node in [15, 25, 50]:
                train_acc, test_acc, train_auc, test_auc = loadLog(f"conv2d_node={node}_size={size}_step={step}_rho=0.100000_dropout={dropout}_epochs=64_num_iter=150_lr=0.001_choose_channel={channel}")
                train_acc_list.append(train_acc.mean())
                test_acc_list.append(test_acc.mean())
                train_auc_list.append(train_auc.mean())
                test_auc_list.append(test_auc.mean())

        pyplot.clf()
        pyplot.figure(figsize=(10, 3))
        pyplot.grid()
        x = np.arange(0.0, 2.5 * len(train_acc_list), 2.5)
        pyplot.bar(x + 0.0, train_acc_list, width=0.5, align="edge", color="r", label="train ACC")
        pyplot.bar(x + 0.5, test_acc_list, width=0.5, align="edge", color="g", label="test ACC")
        pyplot.bar(x + 1.0, train_auc_list, width=0.5, align="edge", color="b", label="train AUC")
        pyplot.bar(x + 1.5, test_auc_list, width=0.5, align="edge", color="c", label="test AUC")
        pyplot.ylim([0.7, 1.0])
        pyplot.xticks(ticks=x + 1, labels=[r"$N_{node} = 15$" + "\nsFC", r"$N_{node} = 25$" + "\nsFC", r"$N_{node} = 50$" + "\nsFC", r"$N_{node} = 15$" + "\n dFC", r"$N_{node} = 25$" + "\n dFC", r"$N_{node} = 50$" + "\n dFC", r"$N_{node} = 15$" + "\n 1 frame", r"$N_{node} = 25$" + "\n 1 frame", r"$N_{node} = 50$" + "\n 1 frame"])
        pyplot.legend(loc="lower right")
        pyplot.savefig(f"bar_channel={channel}_dropout={dropout}.jpg", dpi=720, bbox_inches="tight")


for channel in [1, 2, 4]:
    for dropout in [0.0, 0.1]:
        train_acc_list = []
        test_acc_list = []
        train_auc_list = []
        test_auc_list = []
        for size, step in [(4800, 4800), (480, 180)]:
            for node in [15, 25, 50]:
                dir = f"conv2d_node={node}_size={size}_step={step}_rho=0.100000_dropout={dropout}_epochs=64_num_iter=150_lr=0.001_channel={channel}"
                train_acc = loadFile(os.path.join(dir, "train_acc.log"))[:, -1]
                test_acc = loadFile(os.path.join(dir, "test_acc.log"))[:, -1]
                train_auc = loadFile(os.path.join(dir, "train_auc.log"))[:, -1]
                test_auc = loadFile(os.path.join(dir, "test_auc.log"))[:, -1]
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
                train_auc_list.append(train_auc)
                test_auc_list.append(test_auc)

        pyplot.clf()
        pyplot.figure(figsize=(5, 3))
        pyplot.grid()
        pyplot.boxplot(test_auc_list, vert=False)
        pyplot.yticks(ticks=range(1, 7), labels=[r"$N_{node} = 15$ sFC", r"$N_{node} = 25$ sFC", r"$N_{node} = 50$ sFC", r"$N_{node} = 15$ dFC", r"$N_{node} = 25$ dFC", r"$N_{node} = 50$ dFC"])
        pyplot.savefig(f"test_auc_box_channel={channel}_dropout={dropout}.jpg", dpi=720, bbox_inches="tight")

        pyplot.clf()
        pyplot.figure(figsize=(5, 3))
        pyplot.grid()
        pyplot.boxplot(test_acc_list, vert=False)
        pyplot.yticks(ticks=range(1, 7), labels=[r"$N_{node} = 15$ sFC", r"$N_{node} = 25$ sFC", r"$N_{node} = 50$ sFC", r"$N_{node} = 15$ dFC", r"$N_{node} = 25$ dFC", r"$N_{node} = 50$ dFC"])
        pyplot.savefig(f"test_acc_box_channel={channel}_dropout={dropout}.jpg", dpi=720, bbox_inches="tight")
