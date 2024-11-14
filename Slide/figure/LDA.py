from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

model = LinearDiscriminantAnalysis(n_components=1)
model.fit(x, y)
model.transform
print(model.scalings_)
print(model.xbar_)

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
c = model.xbar_[0]
d = model.xbar_[1]
pyplot.plot([(a * c + b * d + 3.5 * b) / a, (a * c + b * d - 3.5 * b) / a], [-3.5, 3.5], "k--")
pyplot.savefig("lda.jpg", dpi=720, bbox_inches="tight")
