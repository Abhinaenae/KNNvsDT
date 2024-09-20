#By Abhinay Lavu
#09/18/2024
#This program takes in a large dataset and assumes the last column is the label, and all others are features. 
#It then trains models using KNN and Decision trees using the data and displays the differences between the two variants.
################################################################################################################
################################################################################################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Load dataset
df = pd.read_csv('bcw.csv')

# Inspect the first few rows
print(df.head())

# Get descriptive statistics
print(df.info())
print(df.describe())

x = df.drop(columns=['class']) #Features
y = df['class'] #Labels

# Split the dataset (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

scaler = StandardScaler()

# Fit the scaler on the training data, and transform both training and test data
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


#KNN Model Accuracy for a range of K values
for k in [1, 3, 5, 7, 20, 100]:
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(x_train_scaled, y_train)
  knn_predictions = knn.predict(x_test_scaled)
  knn_accuracy = accuracy_score(y_test, knn_predictions)
  print(f"KNN Accuracy with k={k}: {knn_accuracy:.5f}")

print("\n")

#Decision Tree Model Accuracy for a range of max depths
for depth in [1, 3, 5, 7, 20, 100]:
  dt = DecisionTreeClassifier(max_depth=depth)
  dt.fit(x_train, y_train)
  dt_predictions = dt.predict(x_test)
  dt_accuracy = accuracy_score(y_test, dt_predictions)
  print(f"Decision Tree Accuracy with max depth={depth}: {dt_accuracy:.5f}")

print("\n")

#Decision Tree Model Accuracy for a range of min splits
for min_split in [2, 10, 20, 50, 100]:
  dt = DecisionTreeClassifier(min_samples_split=min_split)
  dt.fit(x_train, y_train)
  dt_predictions = dt.predict(x_test)
  accuracy = accuracy_score(y_test, dt_predictions)
  print(f"Decision Tree Accuracy with min samples split={min_split}: {accuracy:.5f}")

print("\n")

#Decision Tree Model Accuracy for a range of min leaves
for min_leaf in [1, 5, 10, 20, 100]:
  dt = DecisionTreeClassifier(min_samples_leaf=min_leaf)
  dt.fit(x_train, y_train)
  dt_predictions = dt.predict(x_test)
  accuracy = accuracy_score(y_test, dt_predictions)
  print(f"Decision Tree Accuracy with min_samples_leaf={min_leaf}: {accuracy:.5f}")

