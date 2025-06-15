import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8)

classifier = GaussianNB()
classifier.fit(x_train, y_train)

probability = classifier.predict_proba(x_test)
y_pred = np.argmax(probability, axis=1)

accuracy = accuracy_score(y_test, y_pred)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap='spring', alpha=0.7, edgecolors='k')
axes[0].set_title("Real labels")

axes[1].scatter(x_test[:, 0], x_test[:, 1], c=y_pred, cmap='spring', alpha=0.7, edgecolors='k')
axes[1].set_title("Predicted labels")

for ax in axes:
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True)

plt.suptitle("Accuracy: %0.3f" % accuracy)
plt.tight_layout()
plt.show()