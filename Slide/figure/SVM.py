import random
import numpy as np
from sklearn import svm
from matplotlib import pyplot

N = 200

np.random.seed(1)
random.seed(1)

x = np.random.randn(N, 2)
y = np.random.randint(0, 2, N)

x[y == 0, 0] += 1.5
x[y == 1, 0] -= 1.5

c = []
for i in y:
    c.append("r" if (i == 0) else 'b')

model = svm.LinearSVC(dual=True)
model.fit(x, y)
print(model.coef_)
print(model.intercept_)

pyplot.clf()
pyplot.figure(figsize=(12, 6))
pyplot.grid()
pyplot.xticks(ticks=[], labels=[])
pyplot.yticks(ticks=[], labels=[])
pyplot.xlim([-5, 5])
pyplot.ylim([-4, 4])
pyplot.scatter(x[:, 0], x[:, 1], c=c)
a = model.coef_[:, 0]
b = model.coef_[:, 1]
c = model.intercept_
pyplot.plot([(c + 3.5 * b) / a, (c - 3.5 * b) / a], [-3.5, 3.5], "k--")
pyplot.savefig("svm.jpg", dpi=720, bbox_inches="tight")
