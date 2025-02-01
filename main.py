import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Display first few rows
print(train_data.head())

# Check for missing values
print(train_data.isnull().sum().sort_values(ascending=False))

# Plot SalePrice distribution
sns.histplot(train_data['SalePrice'], kde=True)
plt.title("Distribution of House Prices")
plt.show()

# Feature selection
selected_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

# Fill missing numerical values with median
num_cols = train_data.select_dtypes(include=['number']).columns
train_data[num_cols] = train_data[num_cols].fillna(train_data[num_cols].median())

# Fill missing categorical values with mode
cat_cols = train_data.select_dtypes(include=['object']).columns
train_data[cat_cols] = train_data[cat_cols].fillna(train_data[cat_cols].mode().iloc[0])

# Encode categorical variables
label_encoders = {}
for col in cat_cols:
    label_encoders[col] = LabelEncoder()
    train_data[col] = label_encoders[col].fit_transform(train_data[col])

# Check feature types
print(train_data[selected_features].dtypes)

# Split data
X = train_data[selected_features]
y = train_data['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Scatter plot of actual vs predicted prices
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()


