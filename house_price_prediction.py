"""
House Price Prediction - Basic Project
Tech: Python, Pandas, Scikit-learn (Linear Regression)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load data
data = pd.read_csv("house_data.csv")
print("Dataset:")
print(data.head())

# Features and target
X = data[["area_sqft", "bedrooms", "age_years"]]
y = data["price_lakhs"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and check accuracy
y_pred = model.predict(X_test)
print("\nModel Accuracy (R2 Score):", round(r2_score(y_test, y_pred), 3))

# Predict price for a new house
new_house = pd.DataFrame({"area_sqft": [1600], "bedrooms": [3], "age_years": [5]})
predicted_price = model.predict(new_house)
print("Predicted price for new house:", round(predicted_price[0], 2), "lakhs")
