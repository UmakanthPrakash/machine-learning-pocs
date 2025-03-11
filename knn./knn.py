# Import necessary libraries
#pip install scikit-learn pandas matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# E1: Load the iris dataset and select only entries of the classes "iris virginica" or "iris versicolor"
# Load the dataset
iris = pd.read_csv('iris.csv')
print(iris.head())
# Filter the dataset to only include "iris virginica" and "iris versicolor"
iris_filtered = iris[iris['Species'].isin(['virginica', 'versicolor'])]

# Separate features (X) and labels (y)
X = iris_filtered.drop('Species', axis=1)
y = iris_filtered['Species']

# E2: Use kNN with K=5 and a train-test-split of 70-30
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the kNN classifier with K=5
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with K=5: {accuracy:.2f}")

# E3: Use extensive search to identify a good k and visualize the accuracy
# Try different values of k
k_values = range(1, 31)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

# Plot the accuracy for different k values
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o')
plt.title('Accuracy vs. k Value')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# Explain the choice of a good k
# The best k is the one that maximizes accuracy. From the plot, we can choose the k with the highest accuracy.
best_k = k_values[accuracies.index(max(accuracies))]
print(f"Best k value: {best_k} with accuracy: {max(accuracies):.2f}")

# Discuss the impact of different distance metrics
# The distance metric can significantly impact kNN performance. Common metrics include Euclidean, Manhattan, and Minkowski.
# Euclidean is the default and works well for most cases, but Manhattan can be better for high-dimensional data.
