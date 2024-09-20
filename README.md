# Analysis and comparison of K-nearest-neighbors and decision tree algorithms for cancer classification
## Introduction
  Given a large dataset consisting of 699 entries of tumors in Wisconsin, we want to classify if a tumor is either cancerous or non-cancerous based on characteristics, such as clump thickness, uniformity of cell size and shape, adhesion of the cells, number of bare nuclei, number of mitotic cells, and more.

## Data preparation
  Two machine learning algorithms are employed to correctly classify the tumors: K-nearest neighbor and Decision Trees. The python script used to calculate the accuracies uses the Sci-Kit learn package for machine learning and the pandas package for processing data. The data was split using train_test_split from sklearn.model_selection using 80% of the data for training and the remaining 20% for testing.
The code below is used to prepare the data before training the models:
```python
x = df.drop(columns=['class']) #Features
y = df['class'] #Labels

# Split the dataset (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

```
## What is K-Nearest-Neighbors?
K-Nearest Neighbors (KNN) is a machine learning algorithm used for both classification and regression tasks. It works by finding the K closest data points to a new input in the feature space, based on some distance metric like Euclidean distance. For classification, it then assigns the most common class among those K neighbors. KNN is non-parametric and instance-based, meaning it doesn't make assumptions about the underlying data distribution and uses the training data directly for predictions.
To train the KNN model, the implementation is performed such that data is scaled to be standardized.
```python
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


#KNN Model Accuracy for a range of K values
for k in [1, 3, 5, 7, 20, 100]:
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(x_train_scaled, y_train)
  knn_predictions = knn.predict(x_test_scaled)
  knn_accuracy = accuracy_score(y_test, knn_predictions)
  print(f"KNN Accuracy with k={k}: {knn_accuracy:.5f}")

```

## What is a Decision Tree?
A decision tree is a tree-like model used for both classification and regression tasks in machine learning. It works by recursively splitting the data based on feature values, creating a flowchart-like structure of decision rules. Each internal node represents a test on a feature, each branch an outcome of that test, and each leaf node a class label or numerical prediction. The tree is built top-down, choosing the best feature to split on at each step according to some criterion like information gain or Gini impurity. Decision trees are intuitive to understand and interpret, but can be prone to overfitting if not properly pruned or regularized.
Multiple models with different hyperparameters are trained, and implemented below:

*Decision Tree Model Accuracy for a range of max depths:*
```python
for depth in [1, 3, 5, 7, 20, 100]:
  dt = DecisionTreeClassifier(max_depth=depth)
  dt.fit(x_train, y_train)
  dt_predictions = dt.predict(x_test)
  dt_accuracy = accuracy_score(y_test, dt_predictions)
  print(f"Decision Tree Accuracy with max depth={depth}: {dt_accuracy:.5f}")
```
*Decision Tree Model Accuracy for a range of min splits:*
```python
for min_split in [2, 10, 20, 50, 100]:
  dt = DecisionTreeClassifier(min_samples_split=min_split)
  dt.fit(x_train, y_train)
  dt_predictions = dt.predict(x_test)
  accuracy = accuracy_score(y_test, dt_predictions)
  print(f"Decision Tree Accuracy with min samples split={min_split}: {accuracy:.5f}")
```
*Decision Tree Model Accuracy for a range of min leaves:*
```python
for min_leaf in [1, 5, 10, 20, 100]:
  dt = DecisionTreeClassifier(min_samples_leaf=min_leaf)
  dt.fit(x_train, y_train)
  dt_predictions = dt.predict(x_test)
  accuracy = accuracy_score(y_test, dt_predictions)
  print(f"Decision Tree Accuracy with min_samples_leaf={min_leaf}: {accuracy:.5f}")
```

## K-Nearest-Neighbors Results
  For the KNN model, I applied a StandardScaler to standardize the features because KNN is a distance-based algorithm. I used 6 different K-values: 1, 3, 5, 7, 20, 100. The results are shown below:

![image](https://github.com/user-attachments/assets/390b493c-1196-4a34-b49c-db530455bf4f)

  A K-value of 1-20 performs at a similar level of 97% accuracy. Having a larger K-value, such as 100, reduces the accuracy of the KNN model.
## Decision Tree Results
  Next, a decision tree model was trained using a range of values with three different hyperparameters: maximum depth of the tree, minimum samples needed for a split, and minimum leaves. No scaling was used since scaling is not necessary for decision trees. The accuracies of each model is shown below:

![image](https://github.com/user-attachments/assets/add1b168-a262-4df0-85c7-11e865352721)

  Using a decision split using a max depth of 3 provides the highest accuracy equivalent to one with the hyperparameter being a high value of minimum samples split, at 95%. Having a maximum depth of 1 and a minimum leaf count of 100 provides the lowest accuracy at 90%. Note that all these accuracies are lower than a KNN algorithm with a low K value.
## Choosing between the two algorithms
  A K-nearest-neighbor algorithm is preferred over a decision tree algorithm for this dataset, especially if the K value is low, such as being the range of [1,20]. This is because this model provides a high accuracy at 97%, compared to a 95% accuracy at best using a decision tree.
