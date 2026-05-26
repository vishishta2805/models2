import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Page Config
st.set_page_config(page_title="Decision Tree Regressor", layout="wide")

st.title("Decision Tree Regressor")

# Dataset
housing = fetch_california_housing()

df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["target"] = housing.target

st.subheader("Dataset")
st.write(df.head())

# Features and Target
X = df.drop("target", axis=1)
y = df["target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Hyperparameters
max_depth = st.slider(
    "Max Depth",
    1,
    20,
    5
)

min_samples_split = st.slider(
    "Min Samples Split",
    2,
    10,
    2
)

# Model
model = DecisionTreeRegressor(
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

st.success(f"MSE: {mse:.2f}")
st.success(f"R2 Score: {r2:.2f}")