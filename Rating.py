import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = 'C:/Users/PMLS/Desktop/python/Internship tasks/codsoft/task3/IMDb Movies India.csv'
df = pd.read_csv(file_path, encoding='latin1')

# Inspect the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Inspect column names
print("Column names in the dataset:")
print(df.columns)

# Clean column names
df.columns = df.columns.str.strip()

# Inspect cleaned column names
print("Cleaned column names in the dataset:")
print(df.columns)

# Inspect the data types and missing values
print("\nData types and missing values:")
print(df.info())

# Summary statistics of the dataset
print("\nSummary statistics:")
print(df.describe())

# Drop duplicates
df = df.drop_duplicates()

# Handle missing values
df = df.dropna(subset=['Rating'])  # Drop rows where Rating is missing

# Convert Duration to numeric values
df['Duration'] = df['Duration'].str.replace(' min', '').astype(float)

# Convert Votes to numeric values
df['Votes'] = df['Votes'].str.replace(',', '').astype(float)

# Fill missing values for categorical features with mode
categorical_features = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
for feature in categorical_features:
    df[feature] = df[feature].fillna(df[feature].mode()[0])

# Fill missing values for numerical features with median
numerical_features = ['Duration', 'Votes']
for feature in numerical_features:
    df[feature] = df[feature].fillna(df[feature].median())

# Verify the cleaning process
print("\nData types and missing values after cleaning:")
print(df.info())

# Plot the distribution of the target variable (Rating)
plt.figure(figsize=(10, 6))
sns.histplot(df['Rating'], bins=20, kde=True)
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# Plot the relationship between Genre and Rating
plt.figure(figsize=(14, 8))
sns.boxplot(x='Genre', y='Rating', data=df)
plt.title('Rating Distribution by Genre')
plt.xticks(rotation=45)
plt.show()

# Plot the relationship between Director and Rating
top_directors = df['Director'].value_counts().nlargest(10).index
plt.figure(figsize=(14, 8))
sns.boxplot(x='Director', y='Rating', data=df[df['Director'].isin(top_directors)])
plt.title('Rating Distribution by Top 10 Directors')
plt.xticks(rotation=45)
plt.show()

# Encode categorical variables
label_encoders = {}
for feature in categorical_features:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature])
    label_encoders[feature] = le

# Create new features (example: count of actors)
df['num_actors'] = df[['Actor 1', 'Actor 2', 'Actor 3']].notna().sum(axis=1)

# Drop irrelevant or redundant columns
df = df.drop(columns=['Name', 'Actor 1', 'Actor 2', 'Actor 3'])

# Verify the feature engineering process
print("\nData after feature engineering:")
print(df.head())

# Split the data into training and testing sets
X = df.drop(columns=['Rating'])
y = df['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate multiple models

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)
print(f"\nLinear Regression - Mean Squared Error: {mse_linear}")
print(f"Linear Regression - R^2 Score: {r2_linear}")

# Support Vector Regression (SVR)
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train, y_train)
y_pred_svr = svr_model.predict(X_test)
mse_svr = mean_squared_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)
print(f"\nSupport Vector Regression - Mean Squared Error: {mse_svr}")
print(f"Support Vector Regression - R^2 Score: {r2_svr}")

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"\nRandom Forest Regressor - Mean Squared Error: {mse_rf}")
print(f"Random Forest Regressor - R^2 Score: {r2_rf}")

# Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)
print(f"\nGradient Boosting Regressor - Mean Squared Error: {mse_gb}")
print(f"Gradient Boosting Regressor - R^2 Score: {r2_gb}")

# Create bar plots and comparison of model performances
models = ['Linear Regression', 'SVR', 'Random Forest', 'Gradient Boosting']
mse_values = [mse_linear, mse_svr, mse_rf, mse_gb]
r2_values = [r2_linear, r2_svr, r2_rf, r2_gb]

plt.figure(figsize=(14, 8))
plt.subplot(1, 2, 1)
sns.barplot(x=models, y=mse_values)
plt.title('Mean Squared Error Comparison')
plt.xlabel('Models')
plt.ylabel('Mean Squared Error')

plt.subplot(1, 2, 2)
sns.barplot(x=models, y=r2_values)
plt.title('R^2 Score Comparison')
plt.xlabel('Models')
plt.ylabel('R^2 Score')

plt.tight_layout()
plt.show()

# Use the best model and try to improve it (Gradient Boosting Regressor)
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_gb_model = grid_search.best_estimator_
print(f"\nBest Gradient Boosting Regressor parameters: {grid_search.best_params_}")

# Evaluate the best model
y_pred_best_gb = best_gb_model.predict(X_test)
mse_best_gb = mean_squared_error(y_test, y_pred_best_gb)
r2_best_gb = r2_score(y_test, y_pred_best_gb)
print(f"\nBest Gradient Boosting Regressor - Mean Squared Error: {mse_best_gb}")
print(f"Best Gradient Boosting Regressor - R^2 Score: {r2_best_gb}")

# Feature Importance from Gradient Boosting
plt.figure(figsize=(10, 6))
sns.barplot(x=best_gb_model.feature_importances_, y=X.columns)
plt.title('Feature Importance (Best Gradient Boosting)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
