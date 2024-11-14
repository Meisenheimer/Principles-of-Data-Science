from tqdm import tqdm
import pandas as pd
import numpy as np


def makeMat(node, data):
    x = np.zeros((node, node))
    k = 0
    for i in range(node):
        for j in range(i + 1, node):
            x[i, j] = x[j, i] = data[k]
            k += 1
    return x


def loadTimeSeries(node, id):
    filename = "../Data/node_timeseries/3T_HCP1200_MSMAll_d" + str(node) + "_ts2/" + str(id) + ".txt"
    fp = open(filename, "r")
    data = fp.readlines()
    fp.close()
    for i in range(len(data)):
        data[i] = [float(item) for item in data[i].split(' ')]
    data = np.array(data)
    data = data.transpose()
    return data


def partial_correlation(ts, rho):
    node = ts.shape[0]
    pcm = np.cov(ts)
    pcm /= np.sqrt((np.mean(pcm.diagonal() ** 2)))
    pcm = np.linalg.inv(pcm + np.abs(rho) * np.eye(node, node))
    if (rho >= 0):
        pcm = -pcm
        tmp = np.sqrt(np.abs(pcm.diagonal()))
        for i in range(node):
            for j in range(node):
                pcm[i, j] /= (tmp[i] * tmp[j])
        for i in range(node):
            pcm[i, i] = 0.0
    return pcm


# def loadData(node, size, step, rho, id):
#     ts = loadTimeSeries(node, id)
#     n, t = ts.shape
#     fc = []
#     for i in range(size - 1, t, step):
#         fc.append(partial_correlation(ts[:, i - size + 1: i + 1], rho))
#     return np.array(fc).reshape(-1, node, node)
#     # fp = open(f"../Data/functional_connectivity/node={node}_size={size}_step={step}_rho={rho:.6f}/{id}.txt", "r")
#     # s = fp.readlines()
#     # fp.close()
#     # for i in range(len(s)):
#     #     s[i] = [float(item) for item in s[i].split(' ')]
#     # return np.array(s).reshape(-1, node, node)


def loadData(node, size, step, rho, id):
    fp = open(f"../Data/functional_connectivity/node={node}_size={size}_step={step}_rho={rho:.6f}/{id}.txt", "r")
    s = fp.readlines()
    fp.close()
    for i in range(len(s)):
        s[i] = [float(item) for item in s[i].split(' ')]
    return np.array(s).reshape(-1, node * (node - 1) // 2)


def loadIdList():
    csv = pd.read_csv("../Data/list.csv")
    data = {}
    for i in range(len(csv)):
        data[int(csv.iloc[i]["Subject"])] = dict(csv.iloc[i])  # Gender: M=1, F=0.
    fp = open("../Data/subjectIDs.txt", "r")
    s = fp.readlines()
    fp.close()
    id_list = []
    for line in tqdm(s):
        id = int(line)
        gender = 1 if data[id]["Gender"] == "M" else 0
        age = data[id]["Age"]
        id_list.append((id, gender, age))
    return id_list


def functional_connectivity(ts, size, step, rho):
    t = ts.shape[1]
    fc = []
    for i in range(size - 1, t, step):
        tmp = ts[:, i - size + 1:i + 1]
        fc.append(partial_correlation(tmp, rho))
    return np.array(fc)


if __name__ == "__main__":
    pass
