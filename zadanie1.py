import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

x, y = make_classification(n_samples=700, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, flip_y=0.08)

np.savetxt("data.csv", np.c_[x, y], delimiter=",", header="f1,f2,label")

plt.figure(figsize=(8, 6))
plt.scatter(x[:, 0], x[:, 1], c=y, cmap="spring", edgecolors="k", alpha=0.7, )
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.title('generated data')
plt.grid(True)
plt.tight_layout()
plt.show()