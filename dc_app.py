import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Page Config
st.set_page_config(page_title="Decision Tree Classifier", layout="wide")

st.title("Decision Tree Classifier")

# Dataset
data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

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
criterion = st.selectbox(
    "Criterion",
    ["gini", "entropy"]
)

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
model = DecisionTreeClassifier(
    criterion=criterion,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

st.success(f"Accuracy: {accuracy:.2f}")