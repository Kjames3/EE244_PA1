# EE244 Computational Learning | Programing Assignment 1

First programming assignment for EE244 Computational Learning

## PA #1: Decision Trees & Random Forests
**Due Date:** April 23, 2026

### Goals
The objective of this Programming Assignment (PA) is to experiment with decision tree and random forest machine learning paradigms using multiple datasets. 

### Assignment Parts
* **(a) Baseline Decision Tree:** Implement a baseline algorithm for induction of decision trees and test it on at least one selected dataset.
* **(b) Pruning and k-NN Comparison:** Incorporate pruning mechanisms, run experiments to observe how pruning affects error rates, and compare results with a k-NN classifier.
* **(c) Learning Curves:** Plot graphs showing induction time and testing error rate versus the number of training examples used (from 10% to 70% of the training set).
* **(d) Random Forest:** Implement a random forest algorithm and evaluate the performance improvement over the best decision tree.

### Datasets Used
* **Mushroom Classification** (for parts a-c)
* **Loan Prediction Dataset** (for part d)

### Environment Setup
A Python virtual environment (`.venv`) has been created with the necessary dependencies installed (`scikit-learn`, `pandas`, `matplotlib`, `pypdf`, etc.).
To run the code:
1. Activate the environment: `.\.venv\Scripts\activate`
2. Run the main script: `python main.py`
