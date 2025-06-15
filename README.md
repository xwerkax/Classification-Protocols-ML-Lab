## Lab 2 – Binary Classification and Experimental Protocols

This repository contains solutions to Lab 2 of a Machine Learning course. The lab focuses on generating synthetic binary classification data, visualizing it, applying a Gaussian Naive Bayes classifier, and evaluating the performance of the model.

## 📋 Tasks Overview

### ✅ Task 1: Data Generation and Visualization
- Used `make_classification` from `scikit-learn` to generate **700 binary classification samples**.
- Only **2 informative features**, with **no redundant or duplicated attributes**.
- Added **8% label noise**.
- Saved the dataset to a `.csv` file using `numpy.savetxt`, with rows containing: feature_1, feature_2, label
- Visualized the dataset using `matplotlib.pyplot.scatter`, with:
- Color-coded classes,
- Grid enabled (`plt.grid()`),
- Tightly laid-out plot (`plt.tight_layout()`).

### ✅ Task 2: Model Training and Evaluation
- Split the dataset using `train_test_split`:
- **80%** training,
- **20%** testing.
- Initialized and trained a **Gaussian Naive Bayes (GaussianNB)** classifier.
- Computed **support probabilities** using `predict_proba`.
- Selected the predicted class as the one with the highest support.
- Evaluated the model using `accuracy_score`.
- Created two subplots:
- Left: true labels,
- Right: predicted labels.
- Included accuracy in the plot title (rounded to 3 decimal places).

## 🧰 Used Libraries
- `scikit-learn`
- `numpy`
- `matplotlib`

## 📁 Files
- `zadanie1.py` – Data generation and visualization
- `zadanie2.py` – Model training, prediction, and evaluation
- `README.md` – This file
