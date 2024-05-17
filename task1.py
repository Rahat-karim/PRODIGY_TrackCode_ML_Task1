# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
train_df = pd.read_csv("train.csv")

# Preprocessing
# Separate numerical and categorical columns
numerical_cols = train_df.select_dtypes(include=['number']).columns
categorical_cols = train_df.select_dtypes(exclude=['number']).columns

# Impute missing values for numerical columns with the mean
imputer = SimpleImputer(strategy='mean')
train_df[numerical_cols] = imputer.fit_transform(train_df[numerical_cols])

# Drop categorical columns for simplicity
train_df = train_df[numerical_cols.tolist() + ['SalePrice']]  # Convert numerical_cols to a list

# Splitting the data into features (X) and target variable (y)
X = train_df.drop(['SalePrice'], axis=1)
y = train_df['SalePrice']

# Splitting the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and fitting the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions on the testing set
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)