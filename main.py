import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
# Replace 'path_to_dataset.csv' with the actual path to your downloaded dataset
data = pd.read_csv('path_to_dataset.csv')

# Data exploration and preprocessing
# Examine the structure and summary statistics of the dataset
print(data.head())
print(data.info())

# Handle missing values
data.dropna(inplace=True)

# Select relevant features and target variable
# Modify the features list based on your dataset's columns
features = ['age', 'country', 'variety', 'region', 'alcohol', 'acidity', 'sugar', 'sulphates']
target = 'price'

# Encode categorical variables
data = pd.get_dummies(data, columns=['country', 'variety', 'region'])

# Split the dataset into training and testing sets
X = data[features]
y = data[target]

# Convert data to NumPy arrays
X = X.to_numpy()
y = y.to_numpy()

# Splitting data into training and testing sets
def train_test_split_custom(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)

# Model training (Random Forest implementation)
class RandomForestRegressorCustom:
    def __init__(self, n_estimators=100, max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.models = [self.create_tree() for _ in range(n_estimators)]
    
    def create_tree(self):
        if self.random_state:
            np.random.seed(self.random_state)
        return {'split_feature': None, 'split_value': None, 'left': None, 'right': None, 'prediction': None}
    
    def fit(self, X, y):
        for i in range(self.n_estimators):
            indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
            self.models[i] = self.build_tree(X[indices], y[indices])
    
    def build_tree(self, X, y):
        node = self.create_tree()
        if len(np.unique(y)) == 1:
            node['prediction'] = np.mean(y)
            return node
        feature_indices = np.random.choice(X.shape[1], size=int(np.sqrt(X.shape[1])), replace=False)
        feature_subset = X[:, feature_indices]
        split_feature = np.random.choice(feature_subset.shape[1], size=1)[0]
        split_value = np.median(feature_subset[:, split_feature])
        left_indices = feature_subset[:, split_feature] <= split_value
        right_indices = ~left_indices
        node['split_feature'] = feature_indices[split_feature]
        node['split_value'] = split_value
        node['left'] = self.build_tree(X[left_indices], y[left_indices])
        node['right'] = self.build_tree(X[right_indices], y[right_indices])
        return node
    
    def predict_single(self, tree, sample):
        if tree['prediction'] is not None:
            return tree['prediction']
        if sample[tree['split_feature']] <= tree['split_value']:
            return self.predict_single(tree['left'], sample)
        else:
            return self.predict_single(tree['right'], sample)
    
    def predict(self, X):
        return np.array([self.predict_single(tree, sample) for sample in X])

model = RandomForestRegressorCustom(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model evaluation (Mean Squared Error)
def mean_squared_error_custom(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

y_pred = model.predict(X_test)
mse = mean_squared_error_custom(y_test, y_pred)
print("Mean Squared Error:", mse)

# Feature importances
def feature_importances_custom(model, feature_names):
    importances = np.zeros(len(feature_names))
    for tree in model.models:
        importances[tree['split_feature']] += 1
    importances /= model.n_estimators
    indices = np.argsort(importances)[::-1]
    return importances, indices

importances, indices = feature_importances_custom(model, features)
top_features = [features[i] for i in indices]

# Visualization of feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(len(top_features)), importances[indices], color="b", align="center")
plt.xticks(range(len(top_features)), top_features, rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

# Interpretation of results
print("Top features influencing wine prices:")
for feature, importance in zip(top_features, importances[indices]):
    print(f"{feature}: {importance:.4f}")
