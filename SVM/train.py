import os
import pandas as pd
import argparse
import random
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn import svm

train_acc_list = []
test_acc_list = []
train_auc_list = []
test_auc_list = []

flog = None


def init(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    return None


def loadIdList(args):
    csv = pd.read_csv(os.path.join(args.data_file_dir, "list.csv"))
    data = {}
    for i in range(len(csv)):
        data[int(csv.iloc[i]["Subject"])] = dict(csv.iloc[i])
    fp = open("../Data/subjectIDs.txt", "r")
    s = fp.readlines()
    fp.close()
    id_list = []
    for line in s:
        id = int(line)
        gender = 1 if data[id]["Gender"] == "M" else 0
        id_list.append((id, gender))
    return id_list


def loadData(args, id_list):
    x = []
    y = []
    for id, gender in id_list:
        fp = open(os.path.join(args.data_dir, "%d.txt" % id), "r")
        s = fp.readlines()
        fp.close()
        for i in range(len(s)):
            s[i] = [float(item) for item in s[i].split(' ')]
        x.append(np.array(s).reshape(-1))
        y.append(gender)
    return x, y


def train(args):
    id_list = {"train": loadIdList(args), "test": []}
    random.shuffle(id_list["train"])
    n = round(len(id_list["train"]) * args.rate)
    id_list["test"] = id_list["train"][:n]
    id_list["train"] = id_list["train"][n:]

    train_x, train_y = loadData(args, id_list["train"])
    test_x, test_y = loadData(args, id_list["test"])

    if (args.Linear):
        model = svm.LinearSVC()
    else:
        model = svm.NuSVC()

    model.fit(train_x, train_y)

    train_pred_y = model.predict(train_x)
    test_pred_y = model.predict(test_x)

    train_acc = accuracy_score(train_y, train_pred_y)
    test_acc = accuracy_score(test_y, test_pred_y)
    train_auc = roc_auc_score(train_y, train_pred_y)
    test_auc = roc_auc_score(test_y, test_pred_y)

    print(f"train acc={train_acc}, test acc={test_acc}, train auc={train_auc}, test auc={test_auc}", file=flog)

    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    train_auc_list.append(train_auc)
    test_auc_list.append(test_auc)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--rate", type=float, default=0.2)

    parser.add_argument("--data_file_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--num_iter", type=int, default=10)

    parser.add_argument("--Linear", type=bool, default=False)

    args = parser.parse_args()
    flog = open(args.output, "w")
    print(args, file=flog)

    for seed in range(0, args.num_iter):
        args.seed = seed
        init(args)
        train(args)

    print(f"train_acc_list: min={np.min(train_acc_list)}, max={np.max(train_acc_list)}, mean={np.mean(train_acc_list)}, var={np.var(train_acc_list)}", file=flog)
    print(f"test_acc_list: min={np.min(test_acc_list)}, max={np.max(test_acc_list)}, mean={np.mean(test_acc_list)}, var={np.var(test_acc_list)}", file=flog)
    print(f"train_auc_list: min={np.min(train_auc_list)}, max={np.max(train_auc_list)}, mean={np.mean(train_auc_list)}, var={np.var(train_auc_list)}", file=flog)
    print(f"test_auc_list: min={np.min(test_auc_list)}, max={np.max(test_auc_list)}, mean={np.mean(test_auc_list)}, var={np.var(test_auc_list)}", file=flog)

    flog.close()
