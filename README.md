# Experiment 4: Support Vector Machine (SVM)

## Aim

To implement and analyze the performance of a Support Vector Machine
(SVM) model for classification using Python.

## Overview

A Support Vector Machine is a supervised machine-learning algorithm used
for classification. It finds a decision boundary (hyperplane) that
separates classes while maximizing the margin between the closest data
points, called **support vectors**.

This experiment uses the **Iris dataset** and compares two SVM kernels:

-   **Linear kernel** -- suitable when classes can be separated
    approximately by a straight decision boundary.
-   **RBF (Radial Basis Function) kernel** -- can model non-linear
    decision boundaries.

For visualization, the experiment uses the first two features of the
Iris dataset.

## Algorithm

1.  **Start.**
2.  Load the Iris dataset and select the first two features for
    visualization.
3.  Split the data into training and testing sets.
4.  Standardize the features using `StandardScaler`.
5.  Train an SVM classifier with a linear kernel.
6.  Predict the test labels and calculate the classification accuracy.
7.  Train another SVM classifier with the RBF kernel.
8.  Predict the test labels and calculate the classification accuracy.
9.  Plot the decision boundaries for both models.
10. **Stop.**

## Program
import matplotlib.pyplot as plt  
import numpy as np        
from sklearn.datasets import load_iris      
from sklearn.metrics import accuracy_score, classification_report  
from sklearn.model_selection import train_test_split    
from sklearn.preprocessing import StandardScaler    
from sklearn.svm import SVC  

\# Load the Iris dataset and select the first two features for visualization  
print("--- Dataset Initialization ---")  
iris = load_iris()  
X = iris.data[:, :2]  # Using first two features (sepal length and sepal width)  
y = iris.target  

print(f"Dataset shape: {X.shape}")    
print("Classes:", iris.target_names)    

\# Split the data into training and testing sets (80:20 ratio)  
X_train, X_test, y_train, y_test = train_test_split(  
    X, y, test_size=0.2, random_state=42  
)  
print(f"Training set size: {X_train.shape[0]} samples")  
print(f"Test set size: {X_test.shape[0]} samples")  

\# Standardize features using StandardScaler  
scaler = StandardScaler()  
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)  

\# Train an SVM classifier with a linear kernel  
print("\n--- Linear Kernel SVM Training & Evaluation ---")  
linear_svm = SVC(kernel="linear", C=1.0, random_state=42)  
linear_svm.fit(X_train_scaled, y_train)    
y_pred_linear = linear_svm.predict(X_test_scaled)  
acc_linear = accuracy_score(y_test, y_pred_linear)  
print(f"Linear Kernel Accuracy: {acc_linear:.4f}")    
print("Classification Report (Linear):\n", classification_report(y_test, y_pred_linear))  
  
\# Train an SVM classifier with the RBF kernel  
print("\n--- RBF Kernel SVM Training & Evaluation ---")    
rbf_svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)  
rbf_svm.fit(X_train_scaled, y_train)          
y_pred_rbf = rbf_svm.predict(X_test_scaled)  
acc_rbf = accuracy_score(y_test, y_pred_rbf)  
print(f"RBF Kernel Accuracy: {acc_rbf:.4f}")  
print("Classification Report (RBF):\n", classification_report(y_test, y_pred_rbf))  

\# Plot the decision boundaries for both models    
print("\n--- Generating Decision Boundary Plots ---")  
fig, axes = plt.subplots(1, 2, figsize=(14, 6))  

x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1  
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1  
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))  

for ax, model, title in zip(  
    axes, [linear_svm, rbf_svm], ["Linear Kernel SVM", "RBF Kernel SVM"]    
):    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])  
    Z = Z.reshape(xx.shape)  
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)  
    ax.scatter(    
        X_train_scaled[:, 0],    
        X_train_scaled[:, 1],  
        c=y_train,  
        cmap=plt.cm.coolwarm,  
        edgecolors="k",    
    )  
    ax.set_title(title)  
    ax.set_xlabel("Sepal Length (Standardized)")  
    ax.set_ylabel("Sepal Width (Standardized)")  
    
plt.tight_layout()  
print("Decision boundary plots generated and rendered successfully.")  
plt.show()  


## output
<img width="578" height="767" alt="image" src="https://github.com/user-attachments/assets/c55c550a-8c0b-417c-9844-cad885d6f713" />
<img width="1397" height="586" alt="image" src="https://github.com/user-attachments/assets/dac2d02a-ed38-4ea2-ae89-6b4efdd40a5d" />


## Result
Thus to implement and analyze the performance of a Support Vector Machine
(SVM) model for classification using Python successfully.
