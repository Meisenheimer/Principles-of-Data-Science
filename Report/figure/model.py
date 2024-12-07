import random
import numpy as np
from sklearn import svm
from matplotlib import pyplot

N = 200

np.random.seed(1)
random.seed(1)

x = np.random.randn(N, 2)
y = np.random.randint(0, 4, N)

x[y == 0, 0] += 1.5
x[y == 1, 0] += 1.5
x[y == 2, 0] -= 1.5
x[y == 3, 0] -= 1.5
x[y == 0, 1] += 1.5
x[y == 1, 1] -= 1.5
x[y == 2, 1] += 1.5
x[y == 3, 1] -= 1.5

c = []
for i in y:
    c.append("r" if (i == 0) else 'b')

tmp = np.zeros(N)
tmp[y == 0] = 1
tmp[y == 1] = 1
model_1 = svm.LinearSVC(dual=True)
model_1.fit(x[tmp == 1], y[tmp == 1])
print(model_1.coef_)
print(model_1.intercept_)

tmp = np.zeros(N)
tmp[y == 0] = 1
tmp[y == 2] = 1
model_2 = svm.LinearSVC(dual=True)
model_2.fit(x[tmp == 1], y[tmp == 1])
print(model_2.coef_)
print(model_2.intercept_)

pyplot.clf()
pyplot.figure(figsize=(6, 6))
pyplot.grid()
pyplot.xticks(ticks=[], labels=[])
pyplot.yticks(ticks=[], labels=[])
pyplot.xlim([-5, 5])
pyplot.ylim([-5, 5])
pyplot.scatter(x[:, 0], x[:, 1], c=c)
a = model_1.coef_[:, 0]
b = model_1.coef_[:, 1]
c = model_1.intercept_
pyplot.plot([-3.5, 3.5], [(c + 3.5 * a) / b, (c - 3.5 * a) / b], "k--")
a = model_2.coef_[:, 0]
b = model_2.coef_[:, 1]
c = model_2.intercept_
pyplot.plot([(c + 3.5 * b) / a, (c - 3.5 * b) / a], [-3.5, 3.5], "k--")
pyplot.savefig("model.jpg", dpi=720, bbox_inches="tight")

x = np.random.randn(N, 2)
y = np.random.randint(0, 4, N)

x[y == 0, 0] += 1.5
x[y == 1, 0] -= 1.5
x[y == 2, 0] -= 1.5
x[y == 3, 0] -= 1.5
x[y == 0, 1] += 0.25
x[y == 1, 1] -= 0.25
x[y == 2, 1] += 0.25
x[y == 3, 1] -= 0.25

c = []
for i in y:
    c.append("r" if (i == 0) else 'b')

tmp = np.zeros(N)
tmp[y == 0] = 1
tmp[y == 2] = 1
model_1 = svm.LinearSVC(dual=True)
model_1.fit(x[tmp == 1], y[tmp == 1])
print(model_1.coef_)
print(model_1.intercept_)

tmp = np.zeros(N)
tmp[y == 0] = 1
tmp[y == 3] = 1
model_2 = svm.LinearSVC(dual=True)
model_2.fit(x[tmp == 1], y[tmp == 1])
print(model_2.coef_)
print(model_2.intercept_)

pyplot.clf()
pyplot.figure(figsize=(6, 6))
pyplot.grid()
pyplot.xticks(ticks=[], labels=[])
pyplot.yticks(ticks=[], labels=[])
pyplot.xlim([-5, 5])
pyplot.ylim([-5, 5])
pyplot.scatter(x[:, 0], x[:, 1], c=c)
a = model_1.coef_[:, 0]
b = model_1.coef_[:, 1]
c = model_1.intercept_
pyplot.plot([-3.5, 3.5], [(c + 3.5 * a) / b, (c - 3.5 * a) / b], "k--")
a = model_2.coef_[:, 0]
b = model_2.coef_[:, 1]
c = model_2.intercept_
pyplot.plot([(c + 3.5 * b) / a, (c - 3.5 * b) / a], [-3.5, 3.5], "k--")
pyplot.savefig("model_worse.jpg", dpi=720, bbox_inches="tight")
