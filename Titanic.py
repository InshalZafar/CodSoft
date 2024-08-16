import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
file_path = "C:/Users/PMLS/Desktop/python/Internship tasks/codsoft/task1/Titanic-Dataset.csv"
titanic_data = pd.read_csv(file_path)

# Initial Data Exploration
print("Initial Data Overview")
print("====================")
print(titanic_data.head())  # Preview the first few rows
print(titanic_data.tail())  # Preview the last few rows
print(titanic_data.describe())  # Summary statistics for all columns
print(titanic_data.info())  # Structure of the dataset
print("\nDimensions of the dataset: ", titanic_data.shape)  # Dimensions of the dataset

# Check for missing values
print("\nMissing Values Per Column")
print("==========================")
missing_values = titanic_data.isnull().sum()
print(missing_values)

# Visualize missing values using seaborn
missing_df = pd.DataFrame({'Feature': missing_values.index, 'MissingValues': missing_values.values})
plt.figure(figsize=(6, 4))
sns.barplot(x='MissingValues', y='Feature', data=missing_df)
plt.title('Missing Values by Feature')
plt.show()

# Data Cleaning and Imputation
# Fill missing 'Age' values
titanic_data['Age'] = titanic_data['Age'].fillna(titanic_data['Age'].median())

# Fill missing 'Embarked' values
titanic_data['Embarked'] = titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0])

# Drop rows with missing 'Fare' or any other remaining NA values
titanic_data.dropna(inplace=True)

# Convert categorical variables to factors
titanic_data['Sex'] = titanic_data['Sex'].astype('category')
titanic_data['Embarked'] = titanic_data['Embarked'].astype('category')
titanic_data['Pclass'] = titanic_data['Pclass'].astype('category')

# Data Exploration after Cleaning
print("\nData Overview After Cleaning")
print("============================")
print(titanic_data.describe())  # Updated summary statistics
print(titanic_data.info())  # Updated structure of the dataset

# Outlier Detection: Boxplot for numeric variables
plt.figure(figsize=(6, 4))
sns.boxplot(x='Pclass', y='Age', data=titanic_data, hue='Pclass', palette='Set3', legend=False)
plt.title('Outlier Detection: Age by Passenger Class')
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x='Survived', y='Fare', data=titanic_data, hue='Survived', palette='Set1', legend=False)
plt.title('Outlier Detection: Fare by Survival Status')
plt.show()

# Visualization 1: Survival Rate by Passenger Class
plt.figure(figsize=(6, 4))
sns.countplot(x='Pclass', hue='Survived', data=titanic_data, palette='RdBu')
plt.title('Survival Rate by Passenger Class')
plt.show()

# Visualization 2: Survival Rate by Gender
plt.figure(figsize=(6, 4))
sns.countplot(x='Sex', hue='Survived', data=titanic_data, palette='Greens')
plt.title('Survival Rate by Gender')
plt.show()

# Visualization 3: Age Distribution with Survival Overlay (Scatter Plot)
plt.figure(figsize=(6, 4))
sns.scatterplot(x='Age', y='Survived', data=titanic_data, hue='Survived', palette='viridis', alpha=0.3)
plt.title('Age Distribution with Survival Overlay (Scatter Plot)')
plt.show()

# Visualization 4: Fare Distribution by Passenger Class (Violin Plot)
plt.figure(figsize=(6, 4))
sns.violinplot(x='Pclass', y='Fare', data=titanic_data)
plt.title('Fare Distribution by Passenger Class (Violin Plot)')
plt.show()

# Ensure the FamilySize feature is created
titanic_data['FamilySize'] = titanic_data['SibSp'] + titanic_data['Parch'] + 1

# Calculate the counts for each FamilySize
family_size_counts = titanic_data['FamilySize'].value_counts().reset_index()
family_size_counts.columns = ['FamilySize', 'Count']

# Visualization 5: Family Size Distribution (Scatter Plot)
plt.figure(figsize=(6, 4))
sns.scatterplot(x='FamilySize', y='Count', data=family_size_counts, color='blue')
plt.title('Family Size Distribution (Scatter Plot)')
plt.show()

# Visualization 6: Survival Rate by Family Size
plt.figure(figsize=(6, 4))
sns.countplot(x='FamilySize', hue='Survived', data=titanic_data, palette='magma')
plt.title('Survival Rate by Family Size')
plt.show()

# Visualization 7: Correlation Matrix for Numeric Features
numeric_features = titanic_data[['Age', 'Fare', 'FamilySize']]
cor_matrix = numeric_features.corr()
print("\nCorrelation Matrix")
print("===================")
print(cor_matrix)
plt.figure(figsize=(6, 4))
sns.heatmap(cor_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Feature Engineering
# Create 'IsAlone' as a new feature
titanic_data['IsAlone'] = np.where(titanic_data['FamilySize'] == 1, 1, 0)

# Drop unnecessary columns
titanic_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Modeling
X = titanic_data.drop('Survived', axis=1)
y = titanic_data['Survived']

# Convert categorical variables to dummy variables
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Logistic Regression
logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train, y_train)
logreg_pred = logreg_model.predict(X_test)
logreg_accuracy = accuracy_score(y_test, logreg_pred)
print("\nLogistic Regression Accuracy: ", logreg_accuracy)

# Model 2: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy: ", rf_accuracy)

# Model Performance Comparison
model_performance = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Accuracy': [logreg_accuracy, rf_accuracy]
})

plt.figure(figsize=(6, 4))
sns.barplot(x='Model', y='Accuracy', data=model_performance)
plt.title('Model Performance Comparison')
plt.ylim(0, 1)
plt.show()

# Generate Confusion Matrices
logreg_cm = confusion_matrix(y_test, logreg_pred)
rf_cm = confusion_matrix(y_test, rf_pred)

# Function to create a confusion matrix heatmap
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Plot Confusion Matrices
plot_confusion_matrix(logreg_cm, 'Confusion Matrix: Logistic Regression')
plot_confusion_matrix(rf_cm, 'Confusion Matrix: Random Forest')
