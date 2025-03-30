"""
You should have studied the SVM algorithm from the lecture and the provided file 'learn-svm.py'. Now, we will apply your SVM knowledge to a dataset. We will create a SVM classifier which predicts whether a person has a breast cancer or not given the features of the person.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2.defaults import KEEP_TRAILING_NEWLINE

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer

# You can find more information about the dataset here:
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
cancer_ds = load_breast_cancer()
df_cancer = pd.DataFrame(np.c_[cancer_ds['data'], cancer_ds['target']],
                         columns = np.append(cancer_ds['feature_names'], ['target']))

# We will do a minimal preprocessing of the dataset
X = df_cancer.drop(['target'], axis=1).to_numpy()
# normalization
X = (X - X.mean()) / X.std()
y = df_cancer['target'].to_numpy()
# convert 0 to -1 for the SVM algorithm
y = np.where(y==0, -1, 1)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

"""
Your job is to complete the following class 'SVM'. 
"""
class SVM:
    def __init__(self, learning_rate=0.0001, n_iters=3000):
        """
        TODO: You should initialize necessary class properties here that includes:
            - learning_rate
            - n_iters (Number of iterations for the optimization algorithm)
            - alphas (Lagrange multipliers)
            - N (Number of samples)
            - K (Kernel function)
            - w (SVM normal vector)
            - b (SVM bias or threshold)
        """
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.alphas = None # Lagrange multipliers
        self.N = None # Number of samples
        self.K = None # Kernel function
        self.w = None # SVM normal vector
        self.b = None # SVM bias or threshold
        self.ZERO = 1e-7

    def fit(self, X, y, C=10):
        """
        TODO: You will use everything you have learned from the lecture and 
          the provided file 'learn-svm.py' to implement the fit method.
        Read the comments and add your code.
        """
        print('Fitting the model...')
        # TODO: Set the number of samples given the input X
        self.N = X.shape[0]
        # TODO: Initialize the Lagrange multipliers alpha
        self.alphas = np.random.rand(self.N) / self.N
        # TODO: Compute the kernel matrix G (i.e., the 2nd term of the target function in the optimization problem)
            # Function
        K = lambda x, y: np.dot(x,y)
            # Initialize empty array
        kernel_matrix = np.zeros((self.N,self.N))
            # Compute matrix
        for i in range(self.N):
            for j in range(self.N):
                kernel_matrix[i,j] = y[i] * y[j] * K(X[i, :], X[j,:])
        # TODO: Optimize Lagrange multipliers iteratively
        losses = []
        for i in range(self.n_iters):
            # Compute Gradient
            dl_alpha = 1 - np.dot(self.alphas, kernel_matrix)
            # Update alphas
            self.alphas += self.learning_rate * dl_alpha
            # Add regularization term
            alphas = np.clip(self.alphas, self.ZERO, C)
            # Compute Loss
            loss = - (np.sum(self.alphas) - 0.5 * np.dot(np.dot(self.alphas, kernel_matrix), self.alphas))
            losses.append(loss)
            
        # TODO: Get w and b, and set the class properties
        # Get w
        m = len(X)
        self.w = np.zeros(X.shape[1])
        for i in range(m):
            self.w = self.w + self.alphas[i] * X[i, :] * y[i]
        # Get b
        C_numeric = C - self.ZERO
        # Get indices of support vectors where alpha < C
        sv_indices = np.where((self.alphas > self.ZERO) & (self.alphas < C_numeric))[0]
        # Average of bias values
        self.b = 0.0
        for ind in sv_indices:
            self.b += y[ind] - np.dot(X[ind, :], self.w)
        
        return self.alphas, losses, self.w, self.b

    def predict(self, X_test):
        """
        This makes predictions using the optimized SVM model.
        Refer to page #32 of the lecture slides for the prediction formula.
        """
        predictions = []
        for i in range(X_test.shape[0]):
            prediction = np.sign(np.dot(self.w, X_test[i, :]) - self.b)
            predictions.append(prediction)
            
        return np.array(predictions)
    
# Create a SVM classifier and train
svm = SVM()
svm.fit(X_train, y_train)
#sv_alphas, sv_losses, sv_w, sv_b = svm.fit(X_train, y_train)

# Evaluate
print(accuracy_score(y_test, svm.predict(X_test)))
