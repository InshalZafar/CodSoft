# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# Load the Iris dataset from the specified path
file_path = 'C:/Users/PMLS/Desktop/python/Internship tasks/codsoft/task2/IRIS.csv'
iris_df = pd.read_csv(file_path)

# Initial Data Inspection
print("First 5 rows of the dataset:")
print(iris_df.head())
print("\nLast 5 rows of the dataset:")
print(iris_df.tail())
print("\nDimensions of the dataset:")
print(iris_df.shape)
print("\nSummary statistics:")
print(iris_df.describe())

# Check for outliers using boxplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.boxplot(x='species', y='sepal_length', data=iris_df, ax=axes[0, 0])
axes[0, 0].set_title('Sepal Length by Species')
sns.boxplot(x='species', y='sepal_width', data=iris_df, ax=axes[0, 1])
axes[0, 1].set_title('Sepal Width by Species')
sns.boxplot(x='species', y='petal_length', data=iris_df, ax=axes[1, 0])
axes[1, 0].set_title('Petal Length by Species')
sns.boxplot(x='species', y='petal_width', data=iris_df, ax=axes[1, 1])
axes[1, 1].set_title('Petal Width by Species')
plt.suptitle('Boxplots of Sepal and Petal Dimensions by Species')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Data Cleaning
# Remove duplicate rows to ensure data integrity
iris_df = iris_df.drop_duplicates()

# Strip any leading/trailing whitespace from the species column
iris_df['species'] = iris_df['species'].str.strip()

# Check for missing or invalid values
print("\nMissing values in each column:")
print(iris_df.isnull().sum())

# Data Exploration
# Pairplot to visualize relationships between features
sns.pairplot(iris_df, hue='species')
plt.suptitle('Pairplot of Iris Dataset', y=1.02)
plt.show()

# Heatmap to visualize correlation between features (excluding the species column)
plt.figure(figsize=(8, 6))
sns.heatmap(iris_df.drop(columns=['species']).corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Iris Dataset')
plt.show()

# Encode the target variable (species) into numeric values
le = LabelEncoder()
iris_df['species'] = le.fit_transform(iris_df['species'])

# Split the data into training and testing sets
X = iris_df.drop('species', axis=1)
y = iris_df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training and Evaluation
# Define the models with important hyperparameters
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\nModel: {name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Hyperparameter Tuning for the Best Model (Random Forest)
# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Perform Grid Search with Cross-Validation
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
print("\nBest Model after Grid Search:")
print(f"Best Model: {best_model}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Feature Importance
# Visualize feature importances for the best model
feature_importances = best_model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features)
plt.title("Feature Importances of Best Model")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

# Decision Boundary Visualization
def plot_decision_boundaries(X, y, model_class, model_name, **model_params):
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Train the model
    model = model_class(**model_params)
    model.fit(X, y)
    
    # Predict the boundaries
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    plt.title(f"Decision Boundary for {model_name}")
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.show()

# Select only the first two features for visualization purposes
X_vis = iris_df[['sepal_length', 'sepal_width']].values
y_vis = iris_df['species']

# Plot decision boundaries for Random Forest
plot_decision_boundaries(X_vis, y_vis, RandomForestClassifier, "Random Forest", n_estimators=100, random_state=42)

# Comprehensive Report
report = f"""
Final Model Report:
Best Model: {best_model}
Accuracy: {accuracy_score(y_test, y_pred):.2f}
Confusion Matrix:
{confusion_matrix(y_test, y_pred)}
Classification Report:
{classification_report(y_test, y_pred)}
"""
print(report)
