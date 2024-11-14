import re
import os
import math
import torch
import Config
from Config import Model, Dataset
from torch import optim
import argparse
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

TOTAL_TIME = 4800
f_train_acc = None
f_train_auc = None
f_test_acc = None
f_test_auc = None
flog = None


def init(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(args, file=flog)

    return None


def calc_loss(output, target):
    return torch.nn.BCELoss()(output, target)


def train(args):
    csv = pd.read_csv(os.path.join(args.data_file_dir, "list.csv"))
    data = {}
    for i in range(len(csv)):
        data[int(csv.iloc[i]["Subject"])] = dict(csv.iloc[i])  # Gender: M=1, F=0.
    fp = open(os.path.join(args.data_file_dir, "subjectIDs.txt"), "r")
    s = fp.readlines()
    fp.close()

    data_list = {"train": [], "test": []}

    for i in tqdm(range(len(s)), file=flog):
        id = int(s[i])
        gender = 1.0 if data[id]["Gender"] == "M" else 0.0
        data_list["train"].append((id, gender))

    random.shuffle(data_list["train"])
    n = round(len(data_list["train"]) * args.rate)
    data_list["test"] = data_list["train"][:n]
    data_list["train"] = data_list["train"][n:]

    train_acc_list = []
    test_acc_list = []

    train_auc_list = []
    test_auc_list = []

    train_loader = Dataset(data_list["train"], args)
    test_loader = Dataset(data_list["test"], args)

    model = Model(args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    bar = tqdm(range(args.epochs), file=flog)
    for epoch in range(args.epochs):
        train_loss = 0.0
        test_loss = 0.0

        model.train()
        model.requires_grad_()
        model.zero_grad()
        cnt = 0.0
        y_true = []
        y_score = []
        for index in range(len(train_loader)):
            mat, target = train_loader[index]
            optimizer.zero_grad()
            output = model(mat)
            y_true.append(True if target > 0.5 else False)
            y_score.append(float(output))
            if ((int(float(output) > 0.5)) == int(target)):
                cnt += 1.0
            loss = calc_loss(output, target)
            loss.backward()
            optimizer.step()
            train_loss += float(loss)
        train_acc_list.append(cnt / float(len(train_loader)))
        train_auc_list.append(roc_auc_score(y_true, y_score))

        model.eval()
        cnt = 0.0
        y_true = []
        y_score = []
        for index in range(len(test_loader)):
            mat, target = test_loader[index]
            output = model(mat)
            y_true.append(True if target > 0.5 else False)
            y_score.append(float(output))
            if ((int(float(output) > 0.5)) == int(target)):
                cnt += 1.0
            loss = calc_loss(output, target)
            test_loss += float(loss)
        test_acc_list.append(cnt / float(len(test_loader)))
        test_auc_list.append(roc_auc_score(y_true, y_score))

        print(test_acc_list[-1], file=f_test_acc, end=" " if epoch != (args.epochs - 1) else "\n")
        print(test_auc_list[-1], file=f_test_auc, end=" " if epoch != (args.epochs - 1) else "\n")
        print(train_acc_list[-1], file=f_train_acc, end=" " if epoch != (args.epochs - 1) else "\n")
        print(train_auc_list[-1], file=f_train_auc, end=" " if epoch != (args.epochs - 1) else "\n")

        bar.set_postfix({"train_loss": train_loss / len(train_loader),
                         "test_loss": test_loss / len(test_loader),
                         "train_acc": train_acc_list[-1],
                         "train_auc": train_auc_list[-1],
                         "test_acc": test_acc_list[-1],
                         "test_auc": test_auc_list[-1]})
        bar.update(1)
    return None


def figure(args, std):
    if std not in ["acc", "auc"]:
        raise
    fp = open(os.path.join(args.output_dir, "train_%s.log" % std), "r")
    train_text = fp.readlines()
    fp.close()
    fp = open(os.path.join(args.output_dir, "test_%s.log" % std), "r")
    test_text = fp.readlines()
    fp.close()

    train_list = np.zeros(args.epochs)
    test_list = np.zeros(args.epochs)

    for i in range(args.num_iter):
        train_list += np.array([float(item) for item in train_text[i].split(' ')])
        test_list += np.array([float(item) for item in test_text[i].split(' ')])

    train_list /= args.num_iter
    test_list /= args.num_iter

    pyplot.clf()
    pyplot.grid()
    pyplot.plot(range(args.epochs), train_list, ".b-", label="train %s" % std)
    pyplot.plot(range(args.epochs), test_list, ".r-", label="test %s" % std)
    pyplot.legend()
    pyplot.title("")
    pyplot.xlabel("Epochs")
    if (std == "acc"):
        pyplot.ylabel("Accuracy")
    elif (std == "auc"):
        pyplot.ylabel("ROC-AUC")
    pyplot.savefig(os.path.join(args.output_dir, "%s.png" % std), dpi=720, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_file_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)

    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--channel", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_iter", type=int, default=150)
    parser.add_argument("--rate", type=float, default=0.2)
    parser.add_argument("--choose", type=bool, default=False)

    args = parser.parse_args()

    re_exp = re.match(r".*node=([0-9]*)_size=([0-9]*)_step=([0-9]*)_rho=([0-9\.]*).*",
                      args.data_dir, re.M | re.I)
    args.node = int(re_exp.group(1))
    args.size = int(re_exp.group(2))
    args.step = int(re_exp.group(3))
    args.rho = float(re_exp.group(4))
    args.input_size = math.floor((TOTAL_TIME - args.size) / args.step) + 1

    os.makedirs(args.output_dir, exist_ok=True)

    Config.flog = flog = open(os.path.join(args.output_dir, "args.log"), "w", encoding="UTF-8")
    f_train_acc = open(os.path.join(args.output_dir, "train_acc.log"), "w", encoding="UTF-8")
    f_train_auc = open(os.path.join(args.output_dir, "train_auc.log"), "w", encoding="UTF-8")
    f_test_acc = open(os.path.join(args.output_dir, "test_acc.log"), "w", encoding="UTF-8")
    f_test_auc = open(os.path.join(args.output_dir, "test_auc.log"), "w", encoding="UTF-8")

    for seed in range(0, args.num_iter):
        args.seed = seed
        init(args)
        train(args)

    flog.close()
    f_train_acc.close()
    f_train_auc.close()
    f_test_acc.close()
    f_test_auc.close()

    figure(args, "acc")
    figure(args, "auc")
