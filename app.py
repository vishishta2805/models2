import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Page Config
st.set_page_config(page_title="KNN Classifier", layout="wide")

st.title("KNN Classifier App")

# Load Dataset
iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

st.subheader("Dataset")
st.write(df.head())

# Features and Target
X = df.drop("target", axis=1)
y = df["target"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter
k = st.slider("Number of Neighbors (K)", 1, 20, 5)

# Model
model = KNeighborsClassifier(n_neighbors=k)

model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

st.success(f"Accuracy: {accuracy:.2f}")